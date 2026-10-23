#include "calibration.h"
#include <QDateTime>
#include <QDir>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPushButton>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>
#include <random>

namespace vs {
static QComboBox *roleBox(const QString &current) {
    auto box = new QComboBox;
    for (auto pair : std::vector<std::pair<QString, QString>>{{"Игрок", "player"},
                                                              {"Либеро", "libero"},
                                                              {"Связующий", "setter"},
                                                              {"Диагональный", "opposite"},
                                                              {"Доигровщик", "outside"},
                                                              {"Центральный", "middle"}})
        box->addItem(pair.first, pair.second);
    box->setCurrentIndex(std::max(0, box->findData(current)));
    return box;
}
static QComboBox *lineupBox(bool active) {
    auto box = new QComboBox;
    box->addItem("На площадке", true);
    box->addItem("Запасной", false);
    box->setCurrentIndex(active ? 0 : 1);
    return box;
}
Canvas::Canvas(QWidget *parent) : QWidget(parent) {
    setMinimumSize(640, 360);
    setMouseTracking(true);
}
QRectF Canvas::imageRect() const {
    if (image.isNull())
        return {};
    QSizeF size = image.size();
    size.scale(this->size(), Qt::KeepAspectRatio);
    return QRectF(QPointF((width() - size.width()) / 2, (height() - size.height()) / 2), size);
}
QPointF Canvas::sourcePoint(QPointF p) const {
    auto r = imageRect();
    return {(p.x() - r.x()) * image.width() / r.width(), (p.y() - r.y()) * image.height() / r.height()};
}
void Canvas::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.fillRect(rect(), QColor(18, 25, 34));
    if (image.isNull())
        return;
    auto r = imageRect();
    p.drawImage(r, image);
    p.save();
    p.translate(r.topLeft());
    double scale = r.width() / image.width();
    p.scale(scale, scale);
    paintOverlay(p, row, geometry, image.size());
    if (editing) {
        p.setBrush(QColor(245, 245, 245));
        p.setPen(QPen(Qt::black, 1 / scale));
        for (auto v : geometry[tool].toArray()) {
            auto a = v.toArray();
            p.drawEllipse(QPointF(a[0].toDouble(), a[1].toDouble()), 5 / scale, 5 / scale);
        }
    }
    p.restore();
}
void Canvas::mousePressEvent(QMouseEvent *event) {
    if (!editing || image.isNull() || !imageRect().contains(event->position()) || tool.isEmpty())
        return;
    auto points = geometry[tool].toArray();
    if (event->button() == Qt::RightButton) {
        if (!points.isEmpty())
            points.removeLast();
        geometry[tool] = points;
        update();
        emit changed();
        return;
    }
    auto point = sourcePoint(event->position());
    double tolerance = 10 * image.width() / imageRect().width();
    for (int i = 0; i < points.size(); ++i) {
        auto a = points[i].toArray();
        if (std::hypot(a[0].toDouble() - point.x(), a[1].toDouble() - point.y()) < tolerance) {
            dragged = i;
            return;
        }
    }
    bool pair = tool == "referee" || tool == "ball" || tool == "player" || tool.startsWith("antenna");
    if (pair && points.size() >= 2)
        points = QJsonArray();
    points.append(QJsonArray{point.x(), point.y()});
    geometry[tool] = points;
    update();
    emit changed();
}
void Canvas::mouseMoveEvent(QMouseEvent *event) {
    if (dragged < 0)
        return;
    auto p = sourcePoint(event->position());
    p.setX(std::clamp(p.x(), 0., double(image.width() - 1)));
    p.setY(std::clamp(p.y(), 0., double(image.height() - 1)));
    auto a = geometry[tool].toArray();
    if (dragged < a.size()) {
        a[dragged] = QJsonArray{p.x(), p.y()};
        geometry[tool] = a;
        update();
        emit changed();
    }
}
void Canvas::mouseReleaseEvent(QMouseEvent *) {
    dragged = -1;
}
Calibration::Calibration(Project &p, double current, QWidget *parent) : QDialog(parent), project(p) {
    setWindowTitle("Площадка и постоянный состав");
    resize(1450, 900);
    auto layout = new QVBoxLayout(this);
    auto body = new QHBoxLayout;
    layout->addLayout(body, 1);
    canvas = new Canvas;
    canvas->editing = true;
    canvas->geometry = project.geometry();
    body->addWidget(canvas, 1);
    auto side = new QVBoxLayout;
    body->addLayout(side);
    tools = new QComboBox;
    for (auto pair : std::vector<std::pair<QString, QString>>{{"court", "Контур видимой площадки"},
                                                              {"center", "Центральная линия"},
                                                              {"attack_a", "Три метра: первая линия"},
                                                              {"attack_b", "Три метра: вторая линия"},
                                                              {"net_top", "Верх сетки"},
                                                              {"net_bottom", "Низ сетки"},
                                                              {"antenna_a", "Левая антенна: низ → верх"},
                                                              {"antenna_b", "Правая антенна: низ → верх"},
                                                              {"referee", "Судья: два угла"},
                                                              {"ball", "Мяч: центр и край"},
                                                              {"player", "Образец игрока: два угла"}})
        tools->addItem(pair.second, pair.first);
    side->addWidget(tools);
    canvas->tool = "court";
    connect(tools, &QComboBox::currentIndexChanged, this, [this] {
        canvas->tool = tools->currentData().toString();
        canvas->update();
    });
    auto clear = new QPushButton("Очистить выбранный элемент");
    side->addWidget(clear);
    connect(clear, &QPushButton::clicked, this, [this] {
        canvas->geometry[canvas->tool] = QJsonArray();
        canvas->update();
    });
    side->addWidget(new QLabel("Линии: несколько точек. Правый клик — убрать точку.\nТочки можно "
                               "перетаскивать. Невидимое не размечайте."));
    nearTeam = new QComboBox;
    nearTeam->addItem("Ближняя команда: A", "A");
    nearTeam->addItem("Ближняя команда: B", "B");
    nearTeam->setCurrentIndex(project.data["nearTeam"].toString("A") == "B" ? 1 : 0);
    side->addWidget(nearTeam);
    mode = new QComboBox;
    mode->addItems({"По внешности, без обязательных номеров", "Внешность + номера на одежде"});
    mode->setCurrentIndex(project.data["numbered"].toBool() ? 1 : 0);
    side->addWidget(mode);
    clothes = new QComboBox;
    clothes->addItems({"Разная одежда", "Общая форма команды (либеро отдельно)"});
    clothes->setCurrentIndex(project.data["uniform"].toBool() ? 1 : 0);
    side->addWidget(clothes);
    farDetail = new QCheckBox("Повышать детализацию дальних игроков");
    farDetail->setChecked(project.data["farDetail"].toBool(true));
    farDetail->setToolTip("Дополнительный проход по дальней части кадра. Отключение ускоряет анализ, но "
                          "может увеличить пропуски людей за сеткой.");
    side->addWidget(farDetail);
    roster = new QTableWidget(0, 7);
    roster->setHorizontalHeaderLabels({"ID", "Команда", "Имя", "Роль", "№", "Образцы", "Состав"});
    roster->setMaximumWidth(540);
    roster->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    roster->setSelectionBehavior(QAbstractItemView::SelectRows);
    side->addWidget(roster, 1);
    for (auto value : project.data["roster"].toArray()) {
        auto o = value.toObject();
        int r = roster->rowCount();
        roster->insertRow(r);
        QStringList values = {QString::number(o["id"].toInt()),
                              o["team"].toString(),
                              o["name"].toString(),
                              o["role"].toString("player"),
                              o["jersey"].toInt(-1) < 0 ? "" : QString::number(o["jersey"].toInt()),
                              QString::number(o["gallery"].toArray().size())};
        for (int c = 0; c < 6; ++c) {
            auto item = new QTableWidgetItem(values[c]);
            if (c == 0 || c == 5)
                item->setFlags(item->flags() & ~Qt::ItemIsEditable);
            roster->setItem(r, c, item);
        }
        roster->setCellWidget(r, 3, roleBox(o["role"].toString("player")));
        roster->setCellWidget(r, 6, lineupBox(o["initiallyActive"].toBool(true)));
    }
    auto buttons = new QHBoxLayout;
    side->addLayout(buttons);
    auto a = new QPushButton("+ Игрок A"), b = new QPushButton("+ Игрок B");
    buttons->addWidget(a);
    buttons->addWidget(b);
    connect(a, &QPushButton::clicked, this, [this] { addPlayer('A'); });
    connect(b, &QPushButton::clicked, this, [this] { addPlayer('B'); });
    auto help =
        new QLabel("Добавьте основных и запасных. Для либеро укажите роль libero.\nВыберите строку, выделите "
                   "игрока двумя углами и сохраните\nобразец. Можно добавить несколько ракурсов одному ID.");
    help->setWordWrap(true);
    side->addWidget(help);
    auto sampleButton = new QPushButton("Сохранить образец выбранного игрока");
    side->addWidget(sampleButton);
    connect(sampleButton, &QPushButton::clicked, this, &Calibration::sample);
    auto footer = new QHBoxLayout;
    layout->addLayout(footer);
    auto prev = new QPushButton("← Кадр"), next = new QPushButton("Кадр →"),
         done = new QPushButton("Сохранить и закрыть");
    footer->addWidget(prev);
    footer->addWidget(next);
    status = new QLabel;
    footer->addWidget(status, 1);
    footer->addWidget(done);
    connect(prev, &QPushButton::clicked, this, [this] {
        if (!loading) {
            persistFrame();
            load(std::max(0, position - 1));
        }
    });
    connect(next, &QPushButton::clicked, this, [this] {
        if (!loading) {
            persistFrame();
            load(std::min(int(times.size()) - 1, position + 1));
        }
    });
    connect(done, &QPushButton::clicked, this, [this] {
        if (!loading) {
            persistFrame();
            if (saveRoster())
                accept();
        }
    });
    times.push_back(std::clamp(current, 0., project.info.times.back()));
    std::mt19937 random(7);
    for (int i = 0; i < 10; ++i) {
        std::uniform_real_distribution<double> distribution(i / 10., (i + 1) / 10.);
        times.push_back(std::min(project.info.times.back(), distribution(random) * project.info.duration));
    }
    load(0);
}
void Calibration::load(int index) {
    loading = true;
    position = index;
    status->setText("Загрузка кадра…");
    QString path = project.video;
    auto frame = std::lower_bound(project.info.times.begin(), project.info.times.end(), times[index]);
    double time = frame == project.info.times.end() ? project.info.times.back() : *frame;
    times[index] = time;
    auto watcher = new QFutureWatcher<QImage>(this);
    connect(watcher, &QFutureWatcher<QImage>::finished, this, [this, watcher, time] {
        loading = false;
        try {
            canvas->setImage(watcher->result());
            canvas->geometry.remove("ball");
            canvas->geometry.remove("player");
            status->setText(
                QString("Кадр %1 / %2 · %3 с").arg(position + 1).arg(times.size()).arg(time, 0, 'f', 2));
        } catch (const std::exception &e) {
            QMessageBox::warning(this, "Кадр", QString::fromUtf8(e.what()));
        }
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run([path, time] { return snapshot(path, time); }));
}
void Calibration::addPlayer(char team) {
    int id = 0;
    for (int r = 0; r < roster->rowCount(); ++r)
        id = std::max(id, roster->item(r, 0)->text().toInt());
    int row = roster->rowCount();
    roster->insertRow(row);
    QStringList values = {QString::number(id + 1), QString(QChar(team)), "", "player", "", "0"};
    for (int c = 0; c < 6; ++c) {
        auto item = new QTableWidgetItem(values[c]);
        if (c == 0 || c == 5)
            item->setFlags(item->flags() & ~Qt::ItemIsEditable);
        roster->setItem(row, c, item);
    }
    roster->selectRow(row);
    roster->setCellWidget(row, 3, roleBox("player"));
    int same = 0;
    for (int r = 0; r < row; ++r)
        if (roster->item(r, 1)->text() == QString(QChar(team)))
            ++same;
    roster->setCellWidget(row, 6, lineupBox(same < 6));
    tools->setCurrentIndex(tools->findData("player"));
}
bool Calibration::saveRoster() {
    try {
        QJsonArray rows;
        auto previous = project.data["roster"].toArray();
        for (int r = 0; r < roster->rowCount(); ++r) {
            int id = roster->item(r, 0)->text().toInt();
            QJsonObject o;
            for (auto v : previous)
                if (v.toObject()["id"].toInt() == id)
                    o = v.toObject();
            QString team = roster->item(r, 1)->text().trimmed().toUpper();
            team.replace(QChar(0x410), "A");
            team.replace(QChar(0x411), "B");
            if (team != "A" && team != "B")
                throw std::runtime_error("Команда должна называться A или B (латиница)");
            o["id"] = id;
            o["team"] = team;
            o["name"] = roster->item(r, 2)->text();
            o["role"] = qobject_cast<QComboBox *>(roster->cellWidget(r, 3))->currentData().toString();
            o["initiallyActive"] =
                qobject_cast<QComboBox *>(roster->cellWidget(r, 6))->currentData().toBool();
            bool ok = false;
            int jersey = roster->item(r, 4)->text().toInt(&ok);
            o["jersey"] = ok ? jersey : -1;
            o["manual"] = true;
            rows.append(o);
        }
        project.data["roster"] = rows;
        project.data["nearTeam"] = nearTeam->currentData().toString();
        project.data["numbered"] = mode->currentIndex() == 1;
        project.data["uniform"] = clothes->currentIndex() == 1;
        project.data["farDetail"] = farDetail->isChecked();
        project.save();
        return true;
    } catch (const std::exception &e) {
        QMessageBox::warning(this, "Состав", QString::fromUtf8(e.what()));
        return false;
    }
}
void Calibration::sample() {
    if (loading || canvas->image.isNull())
        return;
    int r = roster->currentRow();
    if (r < 0) {
        QMessageBox::information(this, "Игрок", "Выберите строку игрока в составе.");
        return;
    }
    auto points = pointsFromJson(canvas->geometry["player"].toArray());
    if (points.size() != 2) {
        tools->setCurrentIndex(tools->findData("player"));
        QMessageBox::information(this, "Образец",
                                 "Отметьте два противоположных угла вокруг одного человека.");
        return;
    }
    double x = std::min(points[0].x, points[1].x), y = std::min(points[0].y, points[1].y),
           w = std::abs(points[0].x - points[1].x), h = std::abs(points[0].y - points[1].y);
    if (w < 10 || h < 25) {
        QMessageBox::warning(this, "Образец", "Выделите человека целиком.");
        return;
    }
    QDir().mkpath(project.folder + "/samples");
    QString name = "samples/" + QString::number(QDateTime::currentMSecsSinceEpoch()) + ".png";
    if (!canvas->image.save(project.folder + "/" + name)) {
        QMessageBox::warning(this, "Образец", "Не удалось сохранить изображение.");
        return;
    }
    int id = roster->item(r, 0)->text().toInt();
    auto samples = project.data["samples"].toArray();
    samples.append(QJsonObject{
        {"player", id}, {"time", times[position]}, {"box", QJsonArray{x, y, w, h}}, {"image", name}});
    project.data["samples"] = samples;
    roster->item(r, 5)->setText(QString::number(roster->item(r, 5)->text().toInt() + 1));
    canvas->geometry.remove("player");
    canvas->update();
    if (!saveRoster())
        return;
    project.data["revision"] = project.data["revision"].toInt() + 1;
    project.save();
    status->setText("Образец добавлен к постоянному ID " + QString::number(id));
}
void Calibration::persistFrame() {
    auto geometry = canvas->geometry;
    auto ball = geometry["ball"].toArray();
    geometry.remove("ball");
    geometry.remove("player");
    project.data["geometry"] = geometry;
    if (ball.size() == 2) {
        auto seeds = project.data["ballSeeds"].toArray();
        seeds.append(QJsonObject{{"time", times[position]}, {"point", ball[0]}});
        project.data["ballSeeds"] = seeds;
    }
    project.data["revision"] = project.data["revision"].toInt() + 1;
    project.save();
}
} // namespace vs
