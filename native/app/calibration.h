#pragma once
#include "project.h"
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QLabel>
#include <QTableWidget>

namespace vs {
class Canvas : public QWidget {
    Q_OBJECT
  public:
    explicit Canvas(QWidget *parent = nullptr);
    QImage image;
    QJsonObject geometry, row;
    QString tool;
    bool editing = false;
    QRectF imageRect() const;
    void setImage(const QImage &im) {
        image = im;
        update();
    }
  signals:
    void changed();

  protected:
    void paintEvent(QPaintEvent *) override;
    void mousePressEvent(QMouseEvent *) override;
    void mouseMoveEvent(QMouseEvent *) override;
    void mouseReleaseEvent(QMouseEvent *) override;

  private:
    int dragged = -1;
    QPointF sourcePoint(QPointF p) const;
};
class Calibration : public QDialog {
    Q_OBJECT
  public:
    explicit Calibration(Project &project, double current, QWidget *parent = nullptr);

  private:
    Project &project;
    Canvas *canvas;
    QComboBox *tools;
    QTableWidget *roster;
    QLabel *status;
    QComboBox *nearTeam;
    QCheckBox *farDetail;
    QComboBox *mode;
    QComboBox *clothes;
    std::vector<double> times;
    int position = 0;
    bool loading = false;
    void load(int index);
    void persistFrame();
    bool saveRoster();
    void addPlayer(char team);
    void sample();
};
} // namespace vs
