#pragma once
#include "models.h"
#include <QMutex>
#include <QThread>
#include <atomic>

namespace vs {
class Worker : public QThread {
    Q_OBJECT
  public:
    Worker(Project project, QString modelFolder, bool exporting = false, QObject *parent = nullptr);
    std::atomic_bool cancelled{false};
    void seek(double time) {
        if (!exporting)
            requested.store(std::max(0., time));
    }
    void confirm(int tracklet, int player, const QImage &portrait);
  signals:
    void status(QString message);
    void progress(double time, int frames, double fps, QJsonObject timings);
    void rosterChanged(QJsonArray roster);
    void uncertain(int tracklet, QJsonArray candidates, QImage portrait);
    void completed(QString output);
    void failed(QString message);

  protected:
    void run() override;

  private:
    Project project;
    QString modelFolder;
    bool exporting;
    std::atomic<double> requested{0};
    QMutex mutex;
    struct Confirmation {
        int tracklet, player;
        QImage portrait;
    };
    std::vector<Confirmation> confirmations;
};
QString cacheContract(const Project &p);
void ensureCache(const Project &p);
QJsonObject cachedFrame(const Project &p, int index);
} // namespace vs
