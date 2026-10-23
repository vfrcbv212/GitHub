#pragma once
#include "core/tracking.h"
#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QPainter>
#include <QString>
#include <memory>

namespace vs {
struct VideoInfo {
    int width = 0, height = 0;
    double fps = 60, duration = 0;
    std::vector<double> times;
};
struct Project {
    QString folder, video;
    QJsonObject data;
    VideoInfo info;
    static Project open(const QString &home, const QString &video);
    void save() const;
    void apply(IdentityTracker &tracker) const;
    QJsonObject geometry() const {
        return data["geometry"].toObject();
    }
};
QString executable(const QString &name);
VideoInfo probe(const QString &path);
QImage snapshot(const QString &path, double time);
QJsonArray featureJson(const Feature &f);
Feature featureFromJson(const QJsonArray &a);
QJsonArray rosterJson(const IdentityTracker &tracker);
void loadRoster(IdentityTracker &tracker, const QJsonArray &a);
QJsonObject observation(const std::vector<Person> &people, const std::optional<Ball> &ball, int index,
                        double time);
void paintOverlay(QPainter &painter, const QJsonObject &row, const QJsonObject &geometry,
                  const QSize &original, bool labels = true);
std::vector<Point> pointsFromJson(const QJsonArray &a);
} // namespace vs
