#pragma once
#include "calibration.h"
#include "worker.h"
#include <QAudioOutput>
#include <QMainWindow>
#include <QMediaPlayer>
#include <QPushButton>
#include <QSettings>
#include <QSlider>
#include <QTimer>
#include <QVideoSink>
#include <optional>

namespace vs {
class Window : public QMainWindow {
    Q_OBJECT
  public:
    Window(QString home, QString models, QWidget *parent = nullptr);
    void openVideo(const QString &file);
    void seek(double seconds);
    bool loaded() const {
        return bool(project);
    }
    int displayedFrame() const {
        return currentFrame;
    }

  protected:
    void closeEvent(QCloseEvent *) override;

  private:
    QString home, modelFolder, outputPath;
    std::optional<Project> project;
    Worker *worker = nullptr;
    Canvas *canvas;
    QMediaPlayer *player;
    QAudioOutput *audio;
    QVideoSink *sink;
    QPushButton *openButton, *calibrateButton, *startButton, *playButton, *stopButton, *exportButton,
        *resultButton;
    QSlider *timeline;
    QComboBox *speed;
    QLabel *status, *timeLabel, *analysisLabel;
    QTimer timer;
    bool wantedPlaying = false, buffering = false, closing = false, busyOpening = false, dialogOpen = false;
    int currentFrame = 0;
    double lastAnalysed = -1, frameTime = -1;
    QSettings settings;
    QJsonObject lastReport;
    bool preciseStill = false;
    int seekGeneration = 0;
    void startAnalysis(bool exporting = false);
    void afterStop(std::function<void()> next);
    void calibrate();
    void togglePlay();
    void updateControls();
    void tick();
    void ask(int tracklet, QJsonArray candidates, QImage portrait);
    void error(const QString &message);
};
} // namespace vs
