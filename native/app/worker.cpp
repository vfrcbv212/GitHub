#include "worker.h"
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QMutexLocker>
#include <QProcess>
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <QUuid>
#include <set>

namespace vs {
QString cacheContract(const Project &p) {
    auto raw = QJsonDocument(QJsonObject{{"source", p.data["sha256"]},
                                         {"geometry", p.data["geometry"]},
                                         {"revision", p.data["revision"]},
                                         {"contract", 4},
                                         {"models", p.data["models"]}})
                   .toJson(QJsonDocument::Compact);
    return QCryptographicHash::hash(raw, QCryptographicHash::Sha256).toHex();
}
static QSqlDatabase database(const Project &p, const QString &name) {
    auto db = QSqlDatabase::contains(name) ? QSqlDatabase::database(name)
                                           : QSqlDatabase::addDatabase("QSQLITE", name);
    db.setDatabaseName(p.folder + "/frames.sqlite");
    if (!db.isOpen() && !db.open())
        throw std::runtime_error(db.lastError().text().toStdString());
    QSqlQuery query(db);
    query.exec("PRAGMA busy_timeout=5000");
    query.exec("PRAGMA journal_mode=WAL");
    return db;
}
void ensureCache(const Project &p) {
    QString name = "setup-" + p.folder;
    {
        auto db = database(p, name);
        QSqlQuery q(db);
        if (!q.exec("CREATE TABLE IF NOT EXISTS meta(contract TEXT)"))
            throw std::runtime_error(q.lastError().text().toStdString());
        q.exec("CREATE TABLE IF NOT EXISTS frames(idx INTEGER PRIMARY KEY,time REAL,payload BLOB)");
        q.exec("SELECT contract FROM meta LIMIT 1");
        bool matches = q.next() && q.value(0).toString() == cacheContract(p);
        if (!matches) {
            q.finish();
            db.transaction();
            q.exec("DELETE FROM frames");
            q.exec("DELETE FROM meta");
            q.prepare("INSERT INTO meta VALUES(?)");
            q.addBindValue(cacheContract(p));
            if (!q.exec() || !db.commit())
                throw std::runtime_error("Не удалось обновить кеш кадров");
        }
        db.close();
    }
    QSqlDatabase::removeDatabase(name);
}
QJsonObject cachedFrame(const Project &p, int index) {
    auto db = database(p, "view-" + p.folder);
    QSqlQuery q(db);
    q.prepare("SELECT payload FROM frames WHERE idx=?");
    q.addBindValue(index);
    if (q.exec() && q.next())
        return QJsonDocument::fromJson(q.value(0).toByteArray()).object();
    return {};
}
Worker::Worker(Project p, QString root, bool output, QObject *parent)
    : QThread(parent), project(std::move(p)), modelFolder(std::move(root)), exporting(output) {}
void Worker::confirm(int tracklet, int player, const QImage &portrait) {
    QMutexLocker lock(&mutex);
    confirmations.push_back({tracklet, player, portrait});
}
static bool readImage(QProcess &decoder, QImage &image, int width, int height, const std::atomic_bool &stop,
                      const std::atomic<double> &seek) {
    qsizetype count = qsizetype(width) * height * 3;
    QByteArray bytes;
    bytes.reserve(count);
    while (bytes.size() < count && !stop.load() && seek.load() < 0) {
        auto data = decoder.read(count - bytes.size());
        if (!data.isEmpty()) {
            bytes += data;
            continue;
        }
        if (decoder.state() == QProcess::NotRunning) {
            data = decoder.read(count - bytes.size());
            bytes += data;
            break;
        }
        decoder.waitForReadyRead(100);
    }
    if (bytes.size() != count)
        return false;
    image = QImage(reinterpret_cast<const uchar *>(bytes.constData()), width, height, width * 3,
                   QImage::Format_RGB888)
                .copy();
    return true;
}
static bool hasNvenc() {
    QProcess test;
    test.start(executable("ffmpeg"), {"-v", "error", "-f", "lavfi", "-i", "color=size=64x64:rate=30", "-t",
                                      "0.1", "-c:v", "h264_nvenc", "-f", "null", "-"});
    if (!test.waitForFinished(7000)) {
        test.kill();
        test.waitForFinished();
        return false;
    }
    return test.exitCode() == 0;
}
void Worker::run() {
    QString connection = "worker-" + QString::number(quintptr(this));
    QString outputPath;
    std::unique_ptr<QProcess> encoder;
    try {
        ensureCache(project);
        auto db = database(project, connection);
        QString runId = QUuid::createUuid().toString();
        QSqlQuery count(db);
        count.exec("SELECT COUNT(*) FROM frames");
        bool completeCache = count.next() && count.value(0).toLongLong() == qint64(project.info.times.size());
        count.finish();
        IdentityTracker tracker;
        project.apply(tracker);
        BallTracker ballTracker;
        std::unique_ptr<Models> models;
        if (!exporting || !completeCache) {
            emit status("Загрузка нативных моделей ONNX…");
            models = std::make_unique<Models>(modelFolder, !project.data["cpuOnly"].toBool());
            if (!models->warning.isEmpty())
                emit status(models->warning);
            // Reviewed appearance samples are explicitly attached to roster cards, including
            // substitutes/liberos.
            for (auto v : project.data["samples"].toArray()) {
                if (cancelled.load())
                    break;
                auto s = v.toObject();
                int id = s["player"].toInt();
                if (!tracker.players.count(id))
                    continue;
                auto b = s["box"].toArray();
                if (b.size() != 4)
                    continue;
                QImage im(project.folder + "/" + s["image"].toString());
                if (im.isNull())
                    continue;
                Person sample;
                sample.box = {b[0].toDouble(), b[1].toDouble(), b[2].toDouble(), b[3].toDouble()};
                models->extract(sample, im, false);
                tracker.players[id].remember(sample.embedding, sample.color);
            }
            emit rosterChanged(rosterJson(tracker));
            emit status("Распознавание: " + models->backend + " · OSNet: CPU · каждый выбранный кадр");
        }
        bool hardwareEncoder = false;
        if (exporting) {
            // Raw RGB pipe has a constant time base: reject variable-rate sources instead of silently
            // changing timing.
            double step = 1 / project.info.fps;
            for (size_t i = 1; i < project.info.times.size(); ++i)
                if (std::abs(project.info.times[i] - project.info.times[i - 1] - step) > .002)
                    throw std::runtime_error("Экспорт этой версии требует постоянной частоты кадров. "
                                             "Просмотр и анализ VFR доступны.");
            outputPath = project.folder + "/annotated-" +
                         QDateTime::currentDateTime().toString("yyyyMMdd-hhmmss-zzz") + ".mp4";
            encoder = std::make_unique<QProcess>();
            hardwareEncoder = hasNvenc();
            QStringList args = {"-v",
                                "error",
                                "-f",
                                "rawvideo",
                                "-pixel_format",
                                "rgb24",
                                "-video_size",
                                QString("%1x%2").arg(project.info.width).arg(project.info.height),
                                "-framerate",
                                QString::number(project.info.fps, 'g', 12),
                                "-i",
                                "-",
                                "-i",
                                project.video,
                                "-map",
                                "0:v:0",
                                "-map",
                                "1:a?",
                                "-c:v",
                                hardwareEncoder ? "h264_nvenc" : "libx264"};
            args << (hardwareEncoder ? QStringList{"-preset", "p4", "-cq", "20"}
                                     : QStringList{"-preset", "veryfast", "-crf", "19"});
            args << "-pix_fmt" << "yuv420p" << "-c:a" << "copy" << "-movflags" << "+faststart" << outputPath;
            encoder->start(executable("ffmpeg"), args);
            if (!encoder->waitForStarted(10000))
                throw std::runtime_error("Не удалось запустить экспорт FFmpeg");
        }
        auto applyConfirmations = [&] {
            std::vector<Confirmation> pending;
            {
                QMutexLocker lock(&mutex);
                pending.swap(confirmations);
            }
            for (const auto &item : pending) {
                if (!tracker.players.count(item.player))
                    continue;
                tracker.assign(item.tracklet, item.player);
                if (models && !item.portrait.isNull()) {
                    Person reviewed;
                    reviewed.box = {0, 0, double(item.portrait.width()), double(item.portrait.height())};
                    models->extract(reviewed, item.portrait, false);
                    tracker.players[item.player].remember(reviewed.embedding, reviewed.color);
                }
                QSqlQuery scan(db), write(db);
                db.transaction();
                scan.prepare("SELECT idx,payload FROM frames WHERE instr(CAST(payload AS TEXT),?)>0");
                scan.addBindValue(runId);
                scan.exec();
                write.prepare("UPDATE frames SET payload=? WHERE idx=?");
                while (scan.next()) {
                    auto row = QJsonDocument::fromJson(scan.value(1).toByteArray()).object();
                    auto people = row["people"].toArray();
                    bool changed = false;
                    for (int k = 0; k < people.size(); ++k) {
                        auto person = people[k].toObject();
                        if (person["tracklet"].toInt() != item.tracklet)
                            continue;
                        auto &card = tracker.players[item.player];
                        person["player"] = item.player;
                        person["team"] = QString(QChar(card.team));
                        person["uncertain"] = false;
                        auto b = person["box"].toArray();
                        bool in =
                            b.size() == 4 && inside(tracker.court, {b[0].toDouble() + b[2].toDouble() / 2,
                                                                    b[1].toDouble() + b[3].toDouble()});
                        person["active"] = card.initiallyActive || in;
                        people[k] = person;
                        changed = true;
                    }
                    if (changed) {
                        row["people"] = people;
                        write.bindValue(0, QJsonDocument(row).toJson(QJsonDocument::Compact));
                        write.bindValue(1, scan.value(0));
                        if (!write.exec())
                            throw std::runtime_error(write.lastError().text().toStdString());
                    }
                }
                scan.finish();
                if (!db.commit())
                    throw std::runtime_error(db.lastError().text().toStdString());
            }
            if (!pending.empty())
                emit rosterChanged(rosterJson(tracker));
        };
        QElapsedTimer elapsed, report, galleryTimer;
        elapsed.start();
        report.start();
        galleryTimer.start();
        int processed = 0;
        bool atEnd = false;
        int sideParity = 0;
        std::set<int> asked;
        double poseTotal = 0, ballTotal = 0, reidTotal = 0, decodeTotal = 0, encodeTotal = 0;
        while (!cancelled.load()) {
            applyConfirmations();
            double target = requested.exchange(-1);
            if (target < 0) {
                if (atEnd) {
                    if (exporting)
                        break;
                    QThread::msleep(30);
                    continue;
                }
                target = 0;
            }
            if (exporting)
                target = 0;
            target = std::clamp(target, 0., project.info.times.back());
            int start =
                int(std::lower_bound(project.info.times.begin(), project.info.times.end(), target - 1e-6) -
                    project.info.times.begin());
            start = std::min(start, int(project.info.times.size()) - 1);
            int decodeStart = std::max(0, start - 8);
            if (models)
                models->resetTemporal();
            ballTracker.reset();
            tracker.resetContinuity();
            atEnd = false;
            elapsed.restart();
            processed = 0;
            poseTotal = ballTotal = reidTotal = decodeTotal = encodeTotal = 0;
            QProcess decoder;
            decoder.start(executable("ffmpeg"),
                          {"-v", "error", "-ss",
                           QString::number(std::max(0., project.info.times[decodeStart] - .00001), 'f', 9),
                           "-i", project.video, "-map", "0:v:0", "-vsync", "0", "-f", "rawvideo", "-pix_fmt",
                           "rgb24", "-threads", "2", "-"});
            if (!decoder.waitForStarted(10000))
                throw std::runtime_error("Не удалось запустить декодирование FFmpeg");
            int index = decodeStart;
            QSqlQuery insert(db);
            insert.prepare("INSERT OR REPLACE INTO frames(idx,time,payload) VALUES(?,?,?)");
            QSqlQuery read(db);
            read.prepare("SELECT payload FROM frames WHERE idx=?");
            while (!cancelled.load() && requested.load() < 0 && index < int(project.info.times.size())) {
                QElapsedTimer stage;
                stage.start();
                std::vector<QImage> images;
                for (int k = 0; k < 9 && index + k < int(project.info.times.size()); ++k) {
                    QImage image;
                    if (!readImage(decoder, image, project.info.width, project.info.height, cancelled,
                                   requested))
                        break;
                    images.push_back(std::move(image));
                }
                decodeTotal += stage.nsecsElapsed() / 1e6;
                if (images.empty())
                    break;
                std::vector<std::vector<BallCandidate>> balls;
                if (models) {
                    balls = models->balls(images);
                    ballTotal += models->ballMs;
                }
                applyConfirmations();
                db.transaction();
                for (size_t j = 0; j < images.size() && !cancelled.load() && requested.load() < 0;
                     ++j, ++index) {
                    double time = project.info.times[index];
                    QJsonObject row;
                    if (models) {
                        auto people = models->poseDetailed(images[j], project.geometry(),
                                                           project.data["farDetail"].toBool(true));
                        poseTotal += models->poseMs;
                        double before = models->reidMs;
                        int parity = 0;
                        for (auto event : project.data["sideChanges"].toArray())
                            if (time >= event.toDouble())
                                parity ^= 1;
                        if (parity != sideParity) {
                            tracker.swapSides();
                            sideParity = parity;
                        }
                        for (auto value : project.data["samples"].toArray()) {
                            auto s = value.toObject();
                            if (std::abs(s["time"].toDouble() - time) > .5 / project.info.fps)
                                continue;
                            auto b = s["box"].toArray();
                            if (b.size() != 4)
                                continue;
                            Box region{b[0].toDouble(), b[1].toDouble(), b[2].toDouble(), b[3].toDouble()};
                            int selected = -1;
                            double best = .35;
                            for (size_t k = 0; k < people.size(); ++k)
                                if (overlap(region, people[k].box) > best) {
                                    best = overlap(region, people[k].box);
                                    selected = int(k);
                                }
                            if (selected >= 0) {
                                people[selected].reviewedPlayer = s["player"].toInt();
                                models->extract(people[selected], images[j], tracker.options.numbered);
                            }
                        }
                        tracker.update(people, time, [&](size_t k) {
                            models->extract(people[k], images[j],
                                            tracker.options.numbered && people[k].readNumber);
                        });
                        reidTotal += models->reidMs - before;
                        std::optional<Point> seed;
                        for (auto v : project.data["ballSeeds"].toArray()) {
                            auto s = v.toObject();
                            if (std::abs(s["time"].toDouble() - time) < .5 / project.info.fps) {
                                auto xy = s["point"].toArray();
                                seed = Point{xy[0].toDouble(), xy[1].toDouble()};
                            }
                        }
                        auto ball = ballTracker.update(balls[j], time, people, seed);
                        row = observation(people, ball, index, time);
                        row["run"] = runId;
                        for (auto &p : people)
                            if (p.uncertain &&
                                (!p.candidates.empty() || inside(tracker.court, p.box.foot())) &&
                                !asked.count(p.tracklet) && time >= target) {
                                asked.insert(p.tracklet);
                                QJsonArray ids;
                                for (int id : p.candidates)
                                    ids.append(id);
                                emit uncertain(p.tracklet, ids,
                                               images[j].copy(QRect(int(p.box.x), int(p.box.y), int(p.box.w),
                                                                    int(p.box.h))));
                            }
                    } else {
                        read.bindValue(0, index);
                        if (!read.exec() || !read.next())
                            throw std::runtime_error("Кеш неполон — повторите анализ");
                        row = QJsonDocument::fromJson(read.value(0).toByteArray()).object();
                        read.finish();
                    }
                    if (index < start)
                        continue;
                    if (models) {
                        insert.bindValue(0, index);
                        insert.bindValue(1, time);
                        insert.bindValue(2, QJsonDocument(row).toJson(QJsonDocument::Compact));
                        if (!insert.exec())
                            throw std::runtime_error(insert.lastError().text().toStdString());
                    }
                    if (encoder) {
                        stage.restart();
                        QImage painted = images[j].convertToFormat(QImage::Format_RGB888);
                        {
                            QPainter p(&painted);
                            paintOverlay(p, row, project.geometry(), painted.size());
                        }
                        QByteArray raw;
                        raw.reserve(painted.width() * painted.height() * 3);
                        for (int y = 0; y < painted.height(); ++y)
                            raw.append(reinterpret_cast<const char *>(painted.constScanLine(y)),
                                       painted.width() * 3);
                        if (encoder->write(raw) != raw.size())
                            throw std::runtime_error("Ошибка записи видео");
                        while (encoder->bytesToWrite() > 8 * 1024 * 1024 && !cancelled.load()) {
                            encoder->waitForBytesWritten(100);
                            if (encoder->state() == QProcess::NotRunning)
                                throw std::runtime_error(encoder->readAllStandardError().toStdString());
                        }
                        encodeTotal += stage.nsecsElapsed() / 1e6;
                    }
                    ++processed;
                    if (report.elapsed() > 120) {
                        emit progress(time, processed, processed / std::max(.001, elapsed.elapsed() / 1000.),
                                      QJsonObject{{"backend", models ? models->backend : "cached"},
                                                  {"pose_ms", poseTotal},
                                                  {"ball_ms", ballTotal},
                                                  {"reid_ms", reidTotal},
                                                  {"decode_ms", decodeTotal},
                                                  {"encode_ms", encodeTotal},
                                                  {"nvenc", hardwareEncoder}});
                        report.restart();
                    }
                }
                if (!db.commit())
                    throw std::runtime_error(db.lastError().text().toStdString());
                if (galleryTimer.elapsed() > 2000) {
                    emit rosterChanged(rosterJson(tracker));
                    galleryTimer.restart();
                }
            }
            decoder.kill();
            decoder.waitForFinished(5000);
            if (models)
                emit rosterChanged(rosterJson(tracker));
            if (requested.load() >= 0 && !exporting)
                continue;
            if (index < int(project.info.times.size()) && !cancelled.load())
                throw std::runtime_error("Декодирование закончилось раньше последнего кадра. " +
                                         decoder.readAllStandardError().toStdString());
            atEnd = true;
            if (!cancelled.load()) {
                emit progress(project.info.times.back(), processed,
                              processed / std::max(.001, elapsed.elapsed() / 1000.),
                              QJsonObject{{"backend", models ? models->backend : "cached"},
                                          {"pose_ms", poseTotal},
                                          {"ball_ms", ballTotal},
                                          {"reid_ms", reidTotal},
                                          {"decode_ms", decodeTotal},
                                          {"encode_ms", encodeTotal},
                                          {"nvenc", hardwareEncoder}});
                if (!exporting)
                    emit completed("");
            }
        }
        if (encoder) {
            encoder->closeWriteChannel();
            if (!encoder->waitForFinished(60000)) {
                encoder->kill();
                encoder->waitForFinished();
                throw std::runtime_error("Экспорт не смог закрыть файл");
            }
            if (encoder->exitCode() != 0 && !cancelled.load())
                throw std::runtime_error(encoder->readAllStandardError().toStdString());
            if (!cancelled.load())
                emit completed(outputPath);
            else {
                QString partial = outputPath;
                partial.replace(".mp4", ".partial.mp4");
                QFile::rename(outputPath, partial);
            }
        }
        db.close();
    } catch (const std::exception &e) {
        if (encoder) {
            encoder->kill();
            encoder->waitForFinished(3000);
        }
        if (!cancelled.load())
            emit failed(QString::fromUtf8(e.what()));
    }
    QSqlDatabase::removeDatabase(connection);
}
} // namespace vs
