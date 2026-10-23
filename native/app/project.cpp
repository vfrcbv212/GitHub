#include "project.h"
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QProcess>
#include <QSaveFile>
#include <QStandardPaths>
#include <stdexcept>

namespace vs {
QString executable(const QString &name) {
#ifdef Q_OS_WIN
    QString filename = name + ".exe";
#else
    QString filename = name;
#endif
    QString local = QCoreApplication::applicationDirPath() + "/" + filename;
    QString isolated = QCoreApplication::applicationDirPath() + "/ffmpeg/" + filename;
    if (QFileInfo::exists(isolated))
        return isolated;
    return QFileInfo::exists(local) ? local : QStandardPaths::findExecutable(filename);
}
static QByteArray run(const QString &name, const QStringList &args, int timeout = 45000) {
    QProcess p;
    QString exe = executable(name);
    if (exe.isEmpty())
        throw std::runtime_error((name + " не найден рядом с EXE").toStdString());
    p.start(exe, args);
    if (!p.waitForStarted(10000) || !p.waitForFinished(timeout)) {
        p.kill();
        p.waitForFinished();
        throw std::runtime_error((name + ": время ожидания истекло").toStdString());
    }
    if (p.exitCode() != 0)
        throw std::runtime_error(p.readAllStandardError().toStdString());
    return p.readAllStandardOutput();
}
VideoInfo probe(const QString &path) {
    auto raw = run("ffprobe",
                   {"-v", "error", "-select_streams", "v:0", "-show_entries",
                    "stream=width,height,avg_frame_rate,duration,nb_frames:stream_tags=rotate:format="
                    "duration:packet=pts_time",
                    "-of", "json", path},
                   120000);
    auto j = QJsonDocument::fromJson(raw).object();
    auto streams = j["streams"].toArray();
    if (streams.isEmpty())
        throw std::runtime_error("В файле нет видеопотока");
    auto s = streams[0].toObject();
    VideoInfo m;
    m.width = s["width"].toInt();
    m.height = s["height"].toInt();
    auto rate = s["avg_frame_rate"].toString().split('/');
    if (rate.size() == 2 && rate[1].toDouble() > 0)
        m.fps = rate[0].toDouble() / rate[1].toDouble();
    m.duration = s["duration"].toString().toDouble();
    if (m.duration <= 0)
        m.duration = j["format"].toObject()["duration"].toString().toDouble();
    for (auto frame : j["packets"].toArray()) {
        QJsonValue value = frame.toObject().value("pts_time");
        if (!value.isUndefined())
            m.times.push_back(value.toString().toDouble());
    }
    if (m.times.empty() || m.width < 2 || m.height < 2)
        throw std::runtime_error("Не удалось прочитать временные метки видео");
    std::sort(m.times.begin(), m.times.end());
    auto expected = s["nb_frames"].toString().toLongLong();
    if (expected > 0 && expected != qint64(m.times.size()))
        throw std::runtime_error("Временные метки пакетов не соответствуют кадрам этого кодека.");
    double origin = m.times.front();
    for (auto &t : m.times)
        t -= origin;
    if (!std::is_sorted(m.times.begin(), m.times.end()))
        throw std::runtime_error("Нарушен порядок временных меток видео");
    return m;
}
QImage snapshot(const QString &path, double time) {
    auto raw = run("ffmpeg", {"-v", "error", "-ss", QString::number(std::max(0., time - .00001), 'f', 9),
                              "-i", path, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"});
    QImage image = QImage::fromData(raw);
    if (image.isNull())
        throw std::runtime_error("Не удалось открыть кадр");
    return image.convertToFormat(QImage::Format_RGB888);
}
Project Project::open(const QString &home, const QString &path) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly))
        throw std::runtime_error("Не удалось открыть видео");
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!f.atEnd())
        hash.addData(f.read(1024 * 1024));
    QString digest = hash.result().toHex();
    Project p;
    p.video = QFileInfo(path).absoluteFilePath();
    p.folder = home + "/projects/" + digest.left(20);
    QDir().mkpath(p.folder);
    QFile saved(p.folder + "/project.json");
    if (saved.open(QIODevice::ReadOnly))
        p.data = QJsonDocument::fromJson(saved.readAll()).object();
    p.info = probe(path);
    p.data["schema"] = 2;
    p.data["sha256"] = digest;
    p.data["video"] = p.video;
    if (!p.data.contains("roster"))
        p.data["roster"] = QJsonArray();
    if (!p.data.contains("nearTeam"))
        p.data["nearTeam"] = "A";
    p.save();
    return p;
}
void Project::save() const {
    QSaveFile file(folder + "/project.json");
    if (!file.open(QIODevice::WriteOnly))
        throw std::runtime_error("Не удалось сохранить проект");
    file.write(QJsonDocument(data).toJson());
    if (!file.commit())
        throw std::runtime_error("Не удалось завершить сохранение проекта");
}
std::vector<Point> pointsFromJson(const QJsonArray &a) {
    std::vector<Point> out;
    for (auto v : a) {
        auto q = v.toArray();
        if (q.size() == 2)
            out.push_back({q[0].toDouble(), q[1].toDouble()});
    }
    return out;
}
QJsonArray featureJson(const Feature &f) {
    QJsonArray a;
    for (float x : f)
        a.append(double(x));
    return a;
}
Feature featureFromJson(const QJsonArray &a) {
    Feature f;
    f.reserve(a.size());
    for (auto v : a)
        f.push_back(float(v.toDouble()));
    return normalized(f);
}
QJsonArray rosterJson(const IdentityTracker &t) {
    QJsonArray out;
    for (auto &[id, p] : t.players) {
        QJsonArray gallery, colors;
        for (auto &f : p.gallery)
            gallery.append(featureJson(f));
        for (auto &f : p.colors)
            colors.append(featureJson(f));
        out.append(QJsonObject{{"id", id},
                               {"team", QString(QChar(p.team))},
                               {"name", QString::fromStdString(p.name)},
                               {"role", QString::fromStdString(p.role)},
                               {"jersey", p.jersey},
                               {"manual", p.manual},
                               {"initiallyActive", p.initiallyActive},
                               {"active", p.active},
                               {"gallery", gallery},
                               {"colors", colors},
                               {"lastSeen", p.lastSeen},
                               {"lastPoint", QJsonArray{p.lastPoint.x, p.lastPoint.y}}});
    }
    return out;
}
void loadRoster(IdentityTracker &t, const QJsonArray &a) {
    for (auto v : a) {
        auto o = v.toObject();
        Player p;
        p.id = o["id"].toInt();
        p.team = o["team"].toString("?").at(0).toLatin1();
        p.name = o["name"].toString().toStdString();
        p.role = o["role"].toString("player").toStdString();
        p.jersey = o["jersey"].toInt(-1);
        p.manual = o["manual"].toBool(true);
        p.initiallyActive = o["initiallyActive"].toBool(true);
        p.active = p.initiallyActive;
        p.lastSeen = o["lastSeen"].toDouble(-1);
        auto pt = o["lastPoint"].toArray();
        if (pt.size() == 2)
            p.lastPoint = {pt[0].toDouble(), pt[1].toDouble()};
        for (auto f : o["gallery"].toArray())
            p.gallery.push_back(featureFromJson(f.toArray()));
        for (auto f : o["colors"].toArray())
            p.colors.push_back(featureFromJson(f.toArray()));
        if (p.id > 0)
            t.players[p.id] = std::move(p);
    }
}
void Project::apply(IdentityTracker &t) const {
    auto g = geometry();
    t.court = pointsFromJson(g["court"].toArray());
    auto c = pointsFromJson(g["center"].toArray());
    if (c.size() >= 2)
        t.center = std::array<Point, 2>{c.front(), c.back()};
    auto r = pointsFromJson(g["referee"].toArray());
    if (r.size() == 2)
        t.referee = Box{std::min(r[0].x, r[1].x), std::min(r[0].y, r[1].y), std::abs(r[1].x - r[0].x),
                        std::abs(r[1].y - r[0].y)};
    t.setNearTeam(data["nearTeam"].toString("A").at(0).toLatin1());
    t.options.uniform = data["uniform"].toBool();
    t.options.numbered = data["numbered"].toBool();
    loadRoster(t, data["roster"].toArray());
    t.options.autoEnroll = t.players.empty();
    double first = 1e99;
    for (auto value : data["samples"].toArray())
        first = std::min(first, value.toObject()["time"].toDouble());
    for (auto value : data["samples"].toArray()) {
        auto sample = value.toObject();
        if (sample["time"].toDouble() > first + 2)
            continue;
        int id = sample["player"].toInt();
        auto b = sample["box"].toArray();
        if (t.players.count(id) && b.size() == 4) {
            char team = t.players.at(id).team;
            for (auto event : data["sideChanges"].toArray())
                if (sample["time"].toDouble() >= event.toDouble())
                    team = team == 'A' ? 'B' : 'A';
            t.calibrate(team, {b[0].toDouble() + b[2].toDouble() * .5, b[1].toDouble() + b[3].toDouble()});
        }
    }
}
QJsonObject observation(const std::vector<Person> &people, const std::optional<Ball> &ball, int index,
                        double time) {
    QJsonArray rows;
    for (const auto &p : people) {
        QJsonArray joints, candidates;
        for (auto &j : p.joints)
            joints.append(QJsonArray{j.p.x, j.p.y, j.score});
        for (int id : p.candidates)
            candidates.append(id);
        rows.append(QJsonObject{{"box", QJsonArray{p.box.x, p.box.y, p.box.w, p.box.h}},
                                {"joints", joints},
                                {"player", p.player},
                                {"tracklet", p.tracklet},
                                {"team", QString(QChar(p.team))},
                                {"referee", p.referee},
                                {"active", p.active},
                                {"uncertain", p.uncertain},
                                {"candidates", candidates}});
    }
    QJsonObject o{{"index", index}, {"time", time}, {"people", rows}};
    if (ball) {
        QJsonArray trail;
        for (auto p : ball->trail)
            trail.append(QJsonArray{p.x, p.y});
        o["ball"] = QJsonObject{{"point", QJsonArray{ball->point.x, ball->point.y}},
                                {"trail", trail},
                                {"uncertain", ball->uncertain}};
    }
    return o;
}
static QColor geometryColor(const QString &key) {
    if (key == "court")
        return QColor(220, 50, 210);
    if (key == "center")
        return QColor(40, 160, 255);
    if (key.startsWith("attack"))
        return QColor(245, 65, 60);
    if (key.startsWith("net"))
        return QColor(245, 220, 35);
    return QColor(80, 245, 110);
}
void paintOverlay(QPainter &p, const QJsonObject &row, const QJsonObject &geometry, const QSize &original,
                  bool labels) {
    p.setRenderHint(QPainter::Antialiasing);
    double width = std::max(1.5, original.width() / 700.);
    p.setFont(QFont("Segoe UI", std::max(10, original.width() / 125)));
    for (auto it = geometry.begin(); it != geometry.end(); ++it) {
        auto v = pointsFromJson(it.value().toArray());
        if (v.size() < 2)
            continue;
        p.setPen(QPen(geometryColor(it.key()), width));
        p.setBrush(Qt::NoBrush);
        QPolygonF polygon;
        for (auto a : v)
            polygon << QPointF(a.x, a.y);
        if (it.key() == "referee" || it.key() == "player")
            p.drawRect(QRectF(polygon[0], polygon[1]).normalized());
        else if (it.key() == "ball")
            p.drawEllipse(polygon[0], distance(v[0], v[1]), distance(v[0], v[1]));
        else if (it.key() == "court")
            p.drawPolygon(polygon);
        else
            p.drawPolyline(polygon);
    }
    static constexpr int bones[][2] = {{5, 6},  {5, 7},   {7, 9},   {6, 8},   {8, 10},  {5, 11},
                                       {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16},
                                       {0, 1},  {0, 2},   {1, 3},   {2, 4}};
    for (auto value : row["people"].toArray()) {
        auto person = value.toObject();
        int id = person["player"].toInt();
        bool referee = person["referee"].toBool();
        auto b = person["box"].toArray();
        if (b.size() != 4)
            continue;
        QColor color = referee                            ? QColor(255, 70, 70)
                       : !id                              ? QColor(165, 165, 165)
                       : person["team"].toString() == "A" ? QColor(255, 200, 70)
                                                          : QColor(70, 175, 255);
        p.setPen(QPen(color, width));
        QRectF box(b[0].toDouble(), b[1].toDouble(), b[2].toDouble(), b[3].toDouble());
        auto court = pointsFromJson(geometry["court"].toArray());
        if (!id && !referee && !court.empty() && !inside(court, {box.center().x(), box.bottom()}))
            continue;
        p.drawRect(box);
        if (labels) {
            QString title = referee ? "Судья"
                            : id ? QString(person["active"].toBool() ? "Игрок %1 / %2" : "Запасной %1 / %2")
                                       .arg(id)
                                       .arg(person["team"].toString())
                            : person["uncertain"].toBool() ? "Личность уточняется"
                                                           : "Человек";
            p.fillRect(QRectF(box.left(), box.top() - 25, p.fontMetrics().horizontalAdvance(title) + 10, 25),
                       QColor(15, 22, 28, 200));
            p.drawText(QPointF(box.left() + 4, box.top() - 6), title);
        }
        auto joints = person["joints"].toArray();
        for (auto bone : bones) {
            if (joints.size() < 17)
                break;
            auto a = joints[bone[0]].toArray(), bq = joints[bone[1]].toArray();
            if (a[2].toDouble() > .3 && bq[2].toDouble() > .3)
                p.drawLine(QPointF(a[0].toDouble(), a[1].toDouble()),
                           QPointF(bq[0].toDouble(), bq[1].toDouble()));
        }
    }
    if (row.contains("ball")) {
        auto b = row["ball"].toObject();
        auto v = pointsFromJson(b["trail"].toArray());
        QPolygonF trail;
        for (auto a : v)
            trail << QPointF(a.x, a.y);
        p.setPen(QPen(QColor(70, 255, 90), width, b["uncertain"].toBool() ? Qt::DashLine : Qt::SolidLine));
        if (trail.size() > 1)
            p.drawPolyline(trail);
        auto xy = b["point"].toArray();
        if (xy.size() == 2)
            p.drawEllipse(QPointF(xy[0].toDouble(), xy[1].toDouble()), 7, 7);
    }
}
} // namespace vs
