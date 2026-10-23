#include "window.h"
#include <QApplication>
#include <QCommandLineParser>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QJsonDocument>
#include <QMessageBox>
#include <QProcess>
#include <QSqlDatabase>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QVideoFrame>
#include <iostream>

using namespace vs;
static void require(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
static int selfTest(const QString &root) {
    QTemporaryDir tmp;
    require(tmp.isValid(), "temporary directory");
    QString source = tmp.path() + "/Проверка — 60 FPS.mp4";
    QProcess ffmpeg;
    ffmpeg.start(executable("ffmpeg"), {"-v",        "error",
                                        "-f",        "lavfi",
                                        "-i",        "testsrc=size=640x360:rate=60",
                                        "-f",        "lavfi",
                                        "-i",        "sine=frequency=440:sample_rate=48000",
                                        "-frames:v", "13",
                                        "-t",        "0.216666667",
                                        "-c:v",      "libx264",
                                        "-pix_fmt",  "yuv420p",
                                        "-c:a",      "aac",
                                        source});
    require(ffmpeg.waitForFinished(30000) && ffmpeg.exitCode() == 0,
            ffmpeg.readAllStandardError().constData());
    std::cerr << "check: media probe/seek\n";
    auto info = probe(source);
    require(info.times.size() == 13, "probe did not preserve all 13 frames");
    require(std::abs(info.times[1] - 1. / 60) < .00001, "original 60 FPS timestamps changed");
    auto image = snapshot(source, .05);
    require(!image.isNull() && image.width() == 640, "Unicode source seek failed");
    std::cerr << "check: ONNX sessions\n";
    Models models(root, false);
    std::cerr << "check: pose inference\n";
    auto people = models.pose(image);
    require(people.size() < 50, "pose output contract invalid");
    Person crop;
    crop.box = {100, 70, 60, 150};
    std::cerr << "check: ReID/OCR inference\n";
    models.extract(crop, image, true);
    require(crop.embedding.size() == 512 && std::abs(cosine(crop.embedding, crop.embedding) - 1) < .001,
            "OSNet did not yield normalized 512D features");
    std::cerr << "check: ball inference\n";
    auto balls = models.balls({image, image, image});
    require(balls.size() == 3, "temporal model lost input frames");
    std::cerr << "check: full native worker\n";
    auto project = Project::open(tmp.path(), source);
    project.data["cpuOnly"] = true;
    project.save();
    Worker worker(project, root);
    QString failure;
    int frames = 0;
    QEventLoop loop;
    QObject::connect(&worker, &Worker::failed, &loop, [&](QString message) {
        failure = message;
        worker.cancelled.store(true);
        loop.quit();
    });
    QObject::connect(&worker, &Worker::progress, &loop,
                     [&](double, int count, double, QJsonObject) { frames = count; });
    QObject::connect(&worker, &Worker::completed, &loop, [&](QString) {
        worker.cancelled.store(true);
        loop.quit();
    });
    QTimer::singleShot(180000, &loop, &QEventLoop::quit);
    worker.start();
    loop.exec();
    worker.cancelled.store(true);
    require(worker.wait(15000), "analysis worker failed to stop");
    require(failure.isEmpty(), failure.toUtf8().constData());
    require(frames == 13, "native analysis did not process every real frame");
    require(cachedFrame(project, 0)["index"].toInt(-1) == 0 &&
                cachedFrame(project, 12)["index"].toInt(-1) == 12,
            "frame cache missing first/last result");
    std::cerr << "check: native export\n";
    Worker exportWorker(project, root, true);
    QString exported;
    QObject::connect(&exportWorker, &Worker::failed, &loop, [&](QString message) {
        failure = message;
        loop.quit();
    });
    QObject::connect(&exportWorker, &Worker::completed, &loop, [&](QString path) {
        exported = path;
        loop.quit();
    });
    QTimer::singleShot(60000, &loop, &QEventLoop::quit);
    exportWorker.start();
    loop.exec();
    exportWorker.cancelled.store(true);
    require(exportWorker.wait(15000), "export worker failed to stop");
    require(failure.isEmpty(), failure.toUtf8().constData());
    require(!exported.isEmpty(), "export did not complete");
    auto output = probe(exported);
    require(output.times.size() == 13, "export dropped or repeated frames");
    require(output.times == info.times, "export changed original CFR timestamps");
    std::cerr << "check: Qt real video playback and seek\n";
    {
        QMediaPlayer media;
        QAudioOutput audio;
        QVideoSink sink;
        media.setAudioOutput(&audio);
        media.setVideoOutput(&sink);
        audio.setVolume(0);
        QEventLoop playback;
        QString mediaError;
        int shown = 0;
        double last = -1;
        auto framesConnection =
            QObject::connect(&sink, &QVideoSink::videoFrameChanged, &playback, [&](const QVideoFrame &frame) {
                if (frame.isValid() && !frame.toImage().isNull()) {
                    ++shown;
                    last = frame.startTime() / 1000000.;
                }
            });
        QObject::connect(&media, &QMediaPlayer::errorOccurred, &playback,
                         [&](QMediaPlayer::Error, const QString &message) {
                             mediaError = message;
                             playback.quit();
                         });
        QObject::connect(&media, &QMediaPlayer::mediaStatusChanged, &playback,
                         [&](QMediaPlayer::MediaStatus state) {
                             if (state == QMediaPlayer::EndOfMedia)
                                 playback.quit();
                         });
        QTimer::singleShot(10000, &playback, &QEventLoop::quit);
        media.setSource(QUrl::fromLocalFile(source));
        media.play();
        playback.exec();
        media.pause();
        require(mediaError.isEmpty(), mediaError.toUtf8().constData());
        require(shown >= 2 && last >= .15, "Qt player did not actually decode and present the video");
        QObject::disconnect(framesConnection);
        QEventLoop seeking;
        bool seeked = false;
        QObject::connect(&sink, &QVideoSink::videoFrameChanged, &seeking, [&](const QVideoFrame &frame) {
            double time = frame.startTime() / 1000000.;
            if (frame.isValid() && time >= .08 && time <= .14 && !frame.toImage().isNull()) {
                seeked = true;
                seeking.quit();
            }
        });
        QTimer::singleShot(10000, &seeking, &QEventLoop::quit);
        media.setPosition(100);
        media.play();
        seeking.exec();
        media.pause();
        require(seeked, "Qt seek did not present the requested part of the video");
        media.setPlaybackRate(10);
        require(std::abs(media.playbackRate() - 10) < .001, "10x playback rate was rejected");
        media.stop();
    }
    std::cerr << "check: Qt GUI\n";
    Window window(tmp.path(), root);
    window.show();
    QApplication::processEvents();
    auto screenshot = window.grab();
    require(!screenshot.isNull(), "native Qt UI did not render");
    window.close();
    std::cout << "PASS native: ONNX pose/temporal ball/512D ReID/OCR, Unicode seek, 13/13 frames, cache, "
                 "full export, original CFR timestamps, Qt real playback/seek/10x, Qt GUI\n";
    return 0;
}
int main(int argc, char **argv) {
    QApplication app(argc, argv);
    app.setApplicationName("VolleyScopeNative");
    app.setOrganizationName("VolleyScope");
    app.setStyle("Fusion");
    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addOption({"home", "Project directory", "path",
                      QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation)});
    parser.addOption(
        {"models", "ONNX model directory", "path", QCoreApplication::applicationDirPath() + "/models"});
    parser.addOption({"self-test", "Run native model/media/export/GUI checks"});
    parser.addOption({"video", "Open a video", "path"});
    parser.process(app);
    try {
        if (parser.isSet("self-test"))
            return selfTest(parser.value("models"));
        Window window(parser.value("home"), parser.value("models"));
        window.show();
        if (parser.isSet("video"))
            window.openVideo(parser.value("video"));
        return app.exec();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        if (!parser.isSet("self-test"))
            QMessageBox::critical(nullptr, "VolleyScope", QString::fromUtf8(e.what()));
        return 1;
    }
}
