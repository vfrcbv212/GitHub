#include "window.h"
#include <QCheckBox>
#include <QCloseEvent>
#include <QCryptographicHash>
#include <QDesktopServices>
#include <QFileDialog>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QMessageBox>
#include <QPointer>
#include <QSaveFile>
#include <QShortcut>
#include <QVBoxLayout>
#include <QVideoFrame>
#include <QtConcurrent/QtConcurrentRun>

namespace vs {
static QString clockText(double seconds) {
    int t = int(std::max(0., seconds));
    return QString("%1:%2").arg(t / 60).arg(t % 60, 2, 10, QChar('0'));
}
Window::Window(QString root, QString models, QWidget *parent)
    : QMainWindow(parent), home(std::move(root)), modelFolder(std::move(models)),
      settings("VolleyScope", "Native") {
    setWindowTitle("VolleyScope Native 0.2 — C++");
    resize(1450, 920);
    auto base = new QWidget;
    setCentralWidget(base);
    auto layout = new QVBoxLayout(base);
    auto toolbar = new QHBoxLayout;
    layout->addLayout(toolbar);
    auto button = [&](QString name) {
        auto b = new QPushButton(name);
        toolbar->addWidget(b);
        return b;
    };
    openButton = button("Открыть видео");
    calibrateButton = button("Площадка и состав");
    startButton = button("Начать анализ");
    stopButton = button("Остановить анализ");
    exportButton = button("Сохранить видео");
    resultButton = button("Открыть результат");
    auto swap = button("Смена сторон");
    auto diagnostics = button("Диагностика");
    connect(diagnostics, &QPushButton::clicked, this, [this] {
        QString name = QFileDialog::getSaveFileName(this, "Сохранить диагностику",
                                                    "VolleyScope-diagnostics.json", "JSON (*.json)");
        if (name.isEmpty())
            return;
        QSaveFile file(name);
        if (!file.open(QIODevice::WriteOnly)) {
            error("Не удалось сохранить диагностику");
            return;
        }
        auto report = lastReport;
        report["version"] = "0.2.0";
        if (project) {
            report["width"] = project->info.width;
            report["height"] = project->info.height;
            report["source_fps"] = project->info.fps;
            report["source_duration"] = project->info.duration;
            report["roster_count"] = project->data["roster"].toArray().size();
        }
        file.write(QJsonDocument(report).toJson());
        file.commit();
    });
    canvas = new Canvas;
    layout->addWidget(canvas, 1);
    auto controls = new QHBoxLayout;
    layout->addLayout(controls);
    auto back = new QPushButton("−5 с"), prev = new QPushButton("◀ Кадр"), next = new QPushButton("Кадр ▶"),
         forward = new QPushButton("+5 с");
    playButton = new QPushButton("▶");
    for (auto b : {back, prev, playButton, next, forward})
        controls->addWidget(b);
    timeline = new QSlider(Qt::Horizontal);
    controls->addWidget(timeline, 1);
    timeLabel = new QLabel("0:00 / 0:00");
    controls->addWidget(timeLabel);
    speed = new QComboBox;
    for (double rate : {.25, .5, 1., 1.5, 2., 4., 6., 8., 10.})
        speed->addItem(QString::number(rate) + "×", rate);
    speed->setCurrentIndex(2);
    controls->addWidget(speed);
    status = new QLabel(
        "Откройте запись. Для постоянных ID один раз заполните состав и добавьте образцы игроков.");
    status->setWordWrap(true);
    layout->addWidget(status);
    analysisLabel = new QLabel(
        "Нативные C++ / ONNX Runtime. Скорость обработки измеряется отдельно от скорости просмотра.");
    analysisLabel->setWordWrap(true);
    layout->addWidget(analysisLabel);
    auto questions = new QCheckBox("Уточнять неоднозначную личность игрока");
    questions->setObjectName("identityQuestions");
    questions->setChecked(settings.value("askIdentities", true).toBool());
    layout->addWidget(questions);
    connect(questions, &QCheckBox::toggled, this,
            [this](bool value) { settings.setValue("askIdentities", value); });
    player = new QMediaPlayer(this);
    audio = new QAudioOutput(this);
    sink = new QVideoSink(this);
    player->setAudioOutput(audio);
    player->setVideoOutput(sink);
    audio->setVolume(.7);
    connect(sink, &QVideoSink::videoFrameChanged, this, [this](const QVideoFrame &frame) {
        if (preciseStill && !wantedPlaying)
            return;
        if (frame.startTime() >= 0)
            frameTime = frame.startTime() / 1000000.;
        auto image = frame.toImage();
        if (!image.isNull())
            canvas->setImage(image);
    });
    connect(player, &QMediaPlayer::errorOccurred, this, [this](QMediaPlayer::Error, const QString &message) {
        if (!message.isEmpty())
            status->setText("Плеер: " + message);
    });
    connect(openButton, &QPushButton::clicked, this, [this] {
        auto file =
            QFileDialog::getOpenFileName(this, "Запись игры", {}, "Видео (*.mp4 *.MP4 *.mov *.MOV *.mkv)");
        if (!file.isEmpty())
            openVideo(file);
    });
    connect(calibrateButton, &QPushButton::clicked, this, &Window::calibrate);
    connect(startButton, &QPushButton::clicked, this, [this] { startAnalysis(); });
    connect(stopButton, &QPushButton::clicked, this, [this] {
        wantedPlaying = false;
        player->pause();
        afterStop([this] { status->setText("Анализ остановлен. Готовые кадры и карточки сохранены."); });
    });
    connect(exportButton, &QPushButton::clicked, this,
            [this] { afterStop([this] { startAnalysis(true); }); });
    connect(resultButton, &QPushButton::clicked, this, [this] {
        if (!outputPath.isEmpty())
            QDesktopServices::openUrl(QUrl::fromLocalFile(outputPath));
    });
    connect(playButton, &QPushButton::clicked, this, &Window::togglePlay);
    connect(timeline, &QSlider::sliderReleased, this, [this] { seek(timeline->value() / 1000.); });
    connect(back, &QPushButton::clicked, this, [this] { seek(player->position() / 1000. - 5); });
    connect(forward, &QPushButton::clicked, this, [this] { seek(player->position() / 1000. + 5); });
    auto step = [this](int direction) {
        if (!project)
            return;
        wantedPlaying = false;
        player->pause();
        int index = std::clamp(currentFrame + direction, 0, int(project->info.times.size()) - 1);
        seek(project->info.times[index]);
    };
    connect(prev, &QPushButton::clicked, this, [step] { step(-1); });
    connect(next, &QPushButton::clicked, this, [step] { step(1); });
    connect(speed, &QComboBox::currentIndexChanged, this,
            [this] { player->setPlaybackRate(speed->currentData().toDouble()); });
    connect(swap, &QPushButton::clicked, this, [this] {
        if (!project)
            return;
        double time = player->position() / 1000.;
        afterStop([this, time] {
            auto events = project->data["sideChanges"].toArray();
            events.append(time);
            project->data["sideChanges"] = events;
            project->data["revision"] = project->data["revision"].toInt() + 1;
            project->save();
            ensureCache(*project);
            startAnalysis();
            worker->seek(time);
            status->setText("Смена сторон сохранена. Команды в карточках игроков сохранены.");
        });
    });
    auto shortcut = [&](const QKeySequence &key, std::function<void()> fn) {
        auto s = new QShortcut(key, this);
        connect(s, &QShortcut::activated, this, std::move(fn));
    };
    shortcut(QKeySequence(Qt::Key_Space), [this] { togglePlay(); });
    shortcut(QKeySequence(Qt::Key_Left), [this] { seek(player->position() / 1000. - 5); });
    shortcut(QKeySequence(Qt::Key_Right), [this] { seek(player->position() / 1000. + 5); });
    shortcut(QKeySequence(Qt::Key_Comma), [step] { step(-1); });
    shortcut(QKeySequence(Qt::Key_Period), [step] { step(1); });
    connect(&timer, &QTimer::timeout, this, &Window::tick);
    timer.start(20);
    updateControls();
}
void Window::error(const QString &message) {
    status->setText(message);
    QMessageBox::warning(this, "VolleyScope", message);
}
void Window::updateControls() {
    bool ready = bool(project) && !busyOpening;
    openButton->setEnabled(!worker && !busyOpening);
    calibrateButton->setEnabled(ready && !worker);
    startButton->setEnabled(ready && !worker);
    stopButton->setEnabled(worker);
    exportButton->setEnabled(ready);
    resultButton->setEnabled(!outputPath.isEmpty());
    playButton->setEnabled(ready);
    timeline->setEnabled(ready);
}
void Window::openVideo(const QString &file) {
    if (worker || busyOpening)
        return;
    busyOpening = true;
    ++seekGeneration;
    preciseStill = false;
    frameTime = -1;
    wantedPlaying = false;
    player->stop();
    updateControls();
    status->setText("Проверка видео и временных меток…");
    auto watcher = new QFutureWatcher<Project>(this);
    QString root = home;
    connect(watcher, &QFutureWatcher<Project>::finished, this, [this, watcher] {
        busyOpening = false;
        try {
            project = watcher->result();
            QFile modelManifest(modelFolder + "/native-models.json");
            if (modelManifest.open(QIODevice::ReadOnly))
                project->data["models"] = QString(
                    QCryptographicHash::hash(modelManifest.readAll(), QCryptographicHash::Sha256).toHex());
            ensureCache(*project);
            canvas->geometry = project->geometry();
            canvas->row = {};
            timeline->setRange(0, int(project->info.duration * 1000));
            player->setSource(QUrl::fromLocalFile(project->video));
            lastAnalysed = -1;
            currentFrame = 0;
            outputPath.clear();
            status->setText(QString("%1×%2 · %3 FPS · %4. Разметка: «Площадка и состав».")
                                .arg(project->info.width)
                                .arg(project->info.height)
                                .arg(project->info.fps, 0, 'f', 3)
                                .arg(clockText(project->info.duration)));
        } catch (const std::exception &e) {
            error(QString::fromUtf8(e.what()));
        }
        updateControls();
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run([root, file] { return Project::open(root, file); }));
}
void Window::afterStop(std::function<void()> next) {
    if (!worker) {
        next();
        return;
    }
    auto old = worker;
    connect(old, &QThread::finished, this, [this, old, next] {
        if (worker == old)
            worker = nullptr;
        old->deleteLater();
        updateControls();
        if (!closing)
            next();
        else
            QTimer::singleShot(0, this, &QWidget::close);
    });
    old->cancelled.store(true);
    status->setText("Завершение текущей операции…");
}
void Window::calibrate() {
    if (!project || worker)
        return;
    try {
        wantedPlaying = false;
        player->pause();
        Calibration dialog(*project, player->position() / 1000., this);
        dialog.exec();
        canvas->geometry = project->geometry();
        ensureCache(*project);
        status->setText("Состав и разметка сохранены. Запустите анализ.");
    } catch (const std::exception &e) {
        error(QString::fromUtf8(e.what()));
    }
}
void Window::startAnalysis(bool exporting) {
    if (!project || worker)
        return;
    try {
        ensureCache(*project);
        worker = new Worker(*project, modelFolder, exporting, this);
        auto active = worker;
        connect(active, &Worker::status, this, [this](const QString &message) { status->setText(message); });
        connect(
            active, &Worker::progress, this, [this](double time, int frames, double fps, QJsonObject timing) {
                lastAnalysed = time;
                lastReport = timing;
                lastReport["frames"] = frames;
                lastReport["processing_fps"] = fps;
                lastReport["source_time"] = time;
                QString backend = timing["backend"].toString();
                analysisLabel->setText(
                    QString(
                        "%1 · %2 кадров · %3 кадров/с обработки · позы %4 мс · мяч %5 мс · ReID/OCR %6 мс")
                        .arg(backend)
                        .arg(frames)
                        .arg(fps, 0, 'f', 1)
                        .arg(timing["pose_ms"].toDouble() / std::max(1, frames), 0, 'f', 1)
                        .arg(timing["ball_ms"].toDouble() / std::max(1, frames), 0, 'f', 1)
                        .arg(timing["reid_ms"].toDouble() / std::max(1, frames), 0, 'f', 1));
            });
        connect(active, &Worker::rosterChanged, this, [this](QJsonArray roster) {
            if (!project)
                return;
            project->data["roster"] = roster;
            try {
                project->save();
            } catch (const std::exception &e) {
                status->setText(QString::fromUtf8(e.what()));
            }
        });
        connect(active, &Worker::uncertain, this, &Window::ask);
        connect(active, &Worker::completed, this, [this, exporting](const QString &path) {
            if (!path.isEmpty()) {
                outputPath = path;
                status->setText("Видео сохранено: " + path);
            } else
                status->setText("Текущий участок проанализирован. Перемотка сохранит карточки игроков.");
            if (exporting)
                updateControls();
        });
        connect(active, &Worker::failed, this, [this, active](const QString &message) {
            wantedPlaying = false;
            player->pause();
            error(message);
            active->cancelled.store(true);
        });
        connect(active, &QThread::finished, this, [this, active] {
            if (worker == active) {
                worker = nullptr;
                active->deleteLater();
                updateControls();
            }
        });
        active->seek(exporting ? 0 : player->position() / 1000.);
        active->start();
        if (!exporting) {
            wantedPlaying = true;
            buffering = true;
            player->pause();
        }
        updateControls();
    } catch (const std::exception &e) {
        error(QString::fromUtf8(e.what()));
    }
}
void Window::togglePlay() {
    if (!project)
        return;
    if (wantedPlaying) {
        wantedPlaying = false;
        player->pause();
    } else {
        wantedPlaying = true;
        preciseStill = false;
        if (!worker && cachedFrame(*project, currentFrame).isEmpty())
            startAnalysis();
        else
            player->play();
    }
    playButton->setText(wantedPlaying ? "Ⅱ" : "▶");
}
void Window::seek(double seconds) {
    if (!project)
        return;
    double target = std::clamp(seconds, 0., project->info.times.back());
    auto selected =
        std::lower_bound(project->info.times.begin(), project->info.times.end(), target - .000001);
    target = selected == project->info.times.end() ? project->info.times.back() : *selected;
    int generation = ++seekGeneration;
    preciseStill = !wantedPlaying;
    if (preciseStill) {
        auto watcher = new QFutureWatcher<QImage>(this);
        QString path = project->video;
        connect(watcher, &QFutureWatcher<QImage>::finished, this, [this, watcher, generation, target] {
            try {
                if (generation == seekGeneration && preciseStill) {
                    canvas->setImage(watcher->result());
                    frameTime = target;
                }
            } catch (const std::exception &e) {
                status->setText(QString::fromUtf8(e.what()));
            }
            watcher->deleteLater();
        });
        watcher->setFuture(QtConcurrent::run([path, target] { return snapshot(path, target); }));
    }
    player->setPosition(qint64(std::round(target * 1000)));
    frameTime = target;
    canvas->row = {};
    if (worker)
        worker->seek(target);
    else if (cachedFrame(*project, int(std::lower_bound(project->info.times.begin(),
                                                        project->info.times.end(), target) -
                                       project->info.times.begin()))
                 .isEmpty()) {
        bool wanted = wantedPlaying;
        startAnalysis();
        if (worker)
            worker->seek(target);
        wantedPlaying = wanted;
    }
    buffering = true;
    if (wantedPlaying)
        player->pause();
}
void Window::tick() {
    if (!project)
        return;
    double time = player->position() / 1000.;
    double exact = frameTime >= 0 ? frameTime : time;
    auto it = std::upper_bound(project->info.times.begin(), project->info.times.end(), exact + .00001);
    currentFrame =
        std::clamp(int(it - project->info.times.begin()) - 1, 0, int(project->info.times.size()) - 1);
    if (!timeline->isSliderDown())
        timeline->setValue(player->position());
    timeLabel->setText(clockText(time) + " / " + clockText(project->info.duration));
    try {
        auto row = cachedFrame(*project, currentFrame);
        canvas->row = row;
        canvas->update();
        if (player->mediaStatus() == QMediaPlayer::EndOfMedia)
            wantedPlaying = false;
        if (wantedPlaying) {
            if (row.isEmpty() && worker) {
                if (!buffering) {
                    buffering = true;
                    player->pause();
                }
                status->setText("Подготовка наложений для выбранного кадра…");
            } else if (buffering || player->playbackState() != QMediaPlayer::PlayingState) {
                buffering = false;
                player->play();
            }
        }
    } catch (const std::exception &e) {
        status->setText(QString::fromUtf8(e.what()));
    }
    playButton->setText(wantedPlaying ? "Ⅱ" : "▶");
}
void Window::ask(int tracklet, QJsonArray candidates, QImage portrait) {
    if (!settings.value("askIdentities", true).toBool() || dialogOpen || !project || !worker)
        return;
    dialogOpen = true;
    auto dialog = new QDialog(this);
    dialog->setWindowTitle("Уточнение личности");
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    auto layout = new QVBoxLayout(dialog);
    auto image = new QLabel;
    image->setPixmap(
        QPixmap::fromImage(portrait).scaled(180, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    layout->addWidget(image);
    layout->addWidget(new QLabel("Кому принадлежит это появление? При сомнении оставьте неизвестным."));
    auto choices = new QComboBox;
    choices->addItem("Пока неизвестно", 0);
    for (auto value : project->data["roster"].toArray()) {
        auto p = value.toObject();
        int id = p["id"].toInt();
        choices->addItem(
            QString("Игрок %1 / %2 · %3%4")
                .arg(id)
                .arg(p["team"].toString(), p["name"].toString(), candidates.contains(id) ? " · похож" : ""),
            id);
    }
    layout->addWidget(choices);
    auto suppress = new QCheckBox("Больше не уточнять автоматически");
    connect(suppress, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked) {
            settings.setValue("askIdentities", false);
            if (auto box = findChild<QCheckBox *>("identityQuestions"))
                box->setChecked(false);
        }
    });
    layout->addWidget(suppress);
    auto apply = new QPushButton("Применить");
    layout->addWidget(apply);
    QPointer<Worker> owner = worker;
    connect(apply, &QPushButton::clicked, dialog,
            [this, dialog, choices, suppress, tracklet, portrait, owner] {
                if (suppress->isChecked()) {
                    settings.setValue("askIdentities", false);
                    if (auto box = findChild<QCheckBox *>("identityQuestions"))
                        box->setChecked(false);
                }
                if (owner && owner == worker && choices->currentData().toInt())
                    owner->confirm(tracklet, choices->currentData().toInt(), portrait);
                dialog->accept();
            });
    connect(dialog, &QDialog::finished, this, [this](int) { dialogOpen = false; });
    dialog->show();
}
void Window::closeEvent(QCloseEvent *event) {
    if (worker) {
        closing = true;
        wantedPlaying = false;
        player->stop();
        event->ignore();
        afterStop([] {});
        return;
    }
    if (busyOpening) {
        event->ignore();
        return;
    }
    timer.stop();
    event->accept();
}
} // namespace vs
