from __future__ import annotations

import copy
import json
import os
import threading
import time
import traceback
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore as C, QtGui as G, QtWidgets as W

from .analytics import MODES, report
from .calibration import LANDMARKS, fit, project_point
from .domain import EVENT_TYPES, STATUSES, VALUE_FIELDS, event, uid, unknown
from .exporting import export_all
from .media import Cancelled, create_proxy, frame_at, frames, import_video
from .store import Project
from .image_io import write_image
from .roster import IDENTITY_MODES,identity_mode,set_identity_mode,player_label

STATUS_LABELS={"confirmed":"Подтверждено человеком","accepted":"Автоматически принято", "observed":"Наблюдение алгоритма","hypothesis":"Предположительно","unknown":"Неизвестно"}
FIELD_LABELS={"type":"Действие","team_id":"Команда","player_id":"Игрок","participants":"Участники блока (список ID)",
    "outcome":"Исход","source_zone":"Исходная зона","target_zone":"Целевая зона","direction":"Направление", "quality":"Качество",
    "target_player_id":"Адресат передачи (ID)","rotation":"Расстановка 1–6","lineup_id":"Состав на интервале (ID)",
    "reception_quality":"Качество предшествующего приёма","blockers":"Количество блокирующих","attack_type":"Тип атаки",
    "floor_point":"Точка контакта пола [x,y] м","block_net_position":"Положение блока вдоль сетки", "initial_direction":"Направление до блока",
    "after_block_direction":"Направление после блока","defender_first_contact":"Первое касание защитника","landing_zone":"Зона падения",
    "credited_player_id":"Индивидуальный автор блокового очка (ID)"}


def combo(items, selected=None):
    widget = W.QComboBox()
    for text, value in items:
        widget.addItem(text, value)
    index = widget.findData(selected)
    if index >= 0: widget.setCurrentIndex(index)
    return widget


def button(text, callback, layout):
    b = W.QPushButton(text); b.clicked.connect(callback); layout.addWidget(b)
    return b


class Signals(C.QObject):
    progress = C.Signal(float, str)
    finished = C.Signal(str)
    failed = C.Signal(str)


class Worker(C.QRunnable):
    def __init__(self, root, fn):
        super().__init__(); self.root=root; self.fn=fn; self.signals=Signals(); self.stop=threading.Event()

    def run(self):
        try:
            with Project(self.root) as p:
                result=self.fn(p,self.signals.progress.emit,self.stop.is_set)
            self.signals.finished.emit(str(result or "Завершено"))
        except Cancelled as exc: self.signals.finished.emit(str(exc))
        except Exception as exc:
            (self.root/"cache"/"last-error.txt").write_text(traceback.format_exc(),encoding="utf-8")
            self.signals.failed.emit(str(exc))


class Canvas(W.QGraphicsView):
    clicked=C.Signal(float,float)
    def __init__(self):
        super().__init__(); self.setScene(W.QGraphicsScene(self)); self.item=W.QGraphicsPixmapItem()
        self.scene().addItem(self.item); self.ratio=1; self.zoomed=False
        self.setMinimumSize(640,360); self.setBackgroundBrush(G.QColor("#17232d"))
        self.setDragMode(W.QGraphicsView.DragMode.ScrollHandDrag)

    def show_frame(self,rgb,overlays=(),points=(),homography=None,geometry_profile=None,polygons=()):
        self.ratio=min(1,1280/rgb.shape[1]); width=round(rgb.shape[1]*self.ratio); height=round(rgb.shape[0]*self.ratio)
        small=cv2.resize(rgb,(width,height)); image=G.QImage(small.data,width,height,small.strides[0],G.QImage.Format.Format_RGB888).copy()
        pix=G.QPixmap.fromImage(image); painter=G.QPainter(pix); painter.setRenderHint(G.QPainter.RenderHint.Antialiasing)
        for d in overlays:
            x1,y1,x2,y2=d["bbox"]; r=self.ratio
            painter.setPen(G.QPen(G.QColor("#67e0de" if d.get("class_id")==1 else "#ffe568"),2))
            painter.drawRect(C.QRectF(x1*r,y1*r,(x2-x1)*r,(y2-y1)*r))
            painter.drawText(C.QPointF(x1*r,max(12,y1*r-4)),d.get("label",d.get("track_id","")))
        painter.setPen(G.QPen(G.QColor("#ff75b0"),3))
        for i,p in enumerate(points):
            painter.drawEllipse(C.QPointF(p[0]*self.ratio,p[1]*self.ratio),4,4)
            painter.drawText(C.QPointF(p[0]*self.ratio+6,p[1]*self.ratio),str(i+1))
        for polygon in polygons:
            vertices=polygon.get("polygon",[])
            if len(vertices)<2:continue
            painter.setPen(G.QPen(G.QColor(polygon.get("color","#81c9f6")),2))
            painter.setBrush(G.QBrush(G.QColor(60,150,210,35)))
            shape=G.QPolygonF([C.QPointF(x*self.ratio,y*self.ratio) for x,y in vertices])
            painter.drawPolygon(shape)
            painter.drawText(shape[0],polygon.get("label",""))
        if homography or (geometry_profile and geometry_profile.get("matrix")):
            inv=np.linalg.inv(np.asarray(homography or geometry_profile["matrix"]))
            painter.setPen(G.QPen(G.QColor("#a1f592"),2,G.Qt.PenStyle.DashLine))
            for a,b in (([0,0],[18,0]),([0,9],[18,9]),([0,0],[0,9]),([18,0],[18,9]),([9,0],[9,9]),([6,0],[6,9]),([12,0],[12,9])):
                world=np.linspace(a,b,25).tolist()
                if geometry_profile:
                    from .calibration import court_to_image
                    pixels=court_to_image(geometry_profile,world,source_id=geometry_profile["source_id"])
                else:pixels=[project_point(inv,p,on_floor=True) for p in world]
                path=G.QPainterPath(C.QPointF(pixels[0][0]*self.ratio,pixels[0][1]*self.ratio))
                for x,y in pixels[1:]:path.lineTo(x*self.ratio,y*self.ratio)
                painter.drawPath(path)
        painter.end(); self.item.setPixmap(pix); self.scene().setSceneRect(pix.rect())
        if not self.zoomed: self.fitInView(self.item,G.Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self,e):
        if e.button()==C.Qt.MouseButton.LeftButton:
            p=self.mapToScene(e.position().toPoint()); self.clicked.emit(p.x()/self.ratio,p.y()/self.ratio)
        super().mousePressEvent(e)

    def wheelEvent(self,e):
        self.zoomed=True; scale=1.2 if e.angleDelta().y()>0 else 1/1.2; self.scale(scale,scale)


class EventDialog(W.QDialog):
    def __init__(self,project,e,parent=None):
        super().__init__(parent); self.setWindowTitle("Событие и достоверность каждого поля"); self.resize(850,700)
        self.value=copy.deepcopy(e); layout=W.QVBoxLayout(self); form=W.QFormLayout(); layout.addLayout(form)
        self.start=W.QDoubleSpinBox(); self.end=W.QDoubleSpinBox()
        for box,value in ((self.start,e["start"]),(self.end,e["end"])):
            box.setRange(0,1e8); box.setDecimals(6); box.setValue(value)
        form.addRow("Время исходника, с",self.start); form.addRow("Конец, с",self.end)
        self.review=combo([("Подтверждено человеком","confirmed"),("Не проверено","unreviewed"),("Отклонено","rejected")],e["review"])
        form.addRow("Проверка события",self.review)
        self.kind=combo([(v,k) for k,v in EVENT_TYPES.items()],e["type"]); form.addRow("Тип",self.kind)
        self.team=combo([("Неизвестна",None)]+[(t["name"],t["id"]) for t in project.all("team")],e.get("team_id")); form.addRow("Команда",self.team)
        self.player=combo([("Неизвестен",None)]+[(f"{p['team_id']} · {player_label(p)}",p["id"]) for p in project.all("player")],e.get("player_id")); form.addRow("Игрок",self.player)
        self.rally=combo([("Неизвестен",None)]+[(f"Розыгрыш {r['id'][:8]} ({r.get('start','?')} с)",r["id"]) for r in project.all("rally")],e.get("rally_id")); form.addRow("Розыгрыш",self.rally)
        layout.addWidget(W.QLabel("Поле пустое = неизвестно. Статус каждого поля независим. accepted означает автоматическое принятие, а не проверку человеком."))
        self.table=W.QTableWidget(len(VALUE_FIELDS),4); self.table.setHorizontalHeaderLabels(["Поле","Значение (списки: JSON)","Достоверность","Причина неизвестности"])
        for i,k in enumerate(VALUE_FIELDS):
            self.table.setItem(i,0,W.QTableWidgetItem(FIELD_LABELS.get(k,k))); v=e.get(k)
            self.table.setItem(i,1,W.QTableWidgetItem("" if v is None else json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else str(v)))
            self.table.setCellWidget(i,2,combo([(STATUS_LABELS[s],s) for s in STATUSES],e.get("evidence",{}).get(k,"unknown")))
            self.table.setItem(i,3,W.QTableWidgetItem(e.get("unknown_reasons",{}).get(k,"")))
        self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch); layout.addWidget(self.table)
        self.floor=W.QCheckBox("Точка floor_point относится к подтверждённому контакту с полом")
        self.floor.setChecked(e.get("context",{}).get("floor_contact_confirmed",False)); layout.addWidget(self.floor)
        box=W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Save|W.QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.save); box.rejected.connect(self.reject); layout.addWidget(box)

    def save(self):
        try:
            e=self.value; e.update(start=self.start.value(),end=self.end.value(),review=self.review.currentData(),rally_id=self.rally.currentData())
            for i,k in enumerate(VALUE_FIELDS):
                text=self.table.item(i,1).text().strip()
                if not text: value=None
                else:
                    try:value=json.loads(text)
                    except json.JSONDecodeError:value=text
                e[k]=value; e["evidence"][k]=self.table.cellWidget(i,2).currentData()
                e["unknown_reasons"][k]=self.table.item(i,3).text()
            for k,value in (("type",self.kind.currentData()),("team_id",self.team.currentData()),("player_id",self.player.currentData())):
                if value!=e.get(k): e["evidence"][k]="confirmed" if value is not None else "unknown"
                e[k]=value
            e.setdefault("context",{})["floor_contact_confirmed"]=self.floor.isChecked()
            from .domain import validate_event
            validate_event(e); self.accept()
        except (ValueError,TypeError) as exc: W.QMessageBox.warning(self,"Проверьте поля",str(exc))


class CalibrationDialog(W.QDialog):
    def __init__(self,project,source,rgb,parent=None,timestamp=0.0):
        super().__init__(parent);self.project=project;self.source=source;self.rgb=rgb;self.timestamp=timestamp
        self.setWindowTitle("Ракурс: ориентиры пола, приблизительные зоны и объектив");self.resize(1250,950)
        from .calibration import validate_profile
        prior=[p for p in project.all("calibration") if p["source_id"]==source["id"]]
        self.previous={}
        for candidate in reversed(prior):
            if candidate.get("start",0)<=timestamp and (candidate.get("end") is None or timestamp<=candidate["end"]):
                try:
                    self.previous=validate_profile(candidate,source_id=source["id"],source_sha256=source["sha256"],image_size=[rgb.shape[1],rgb.shape[0]])
                    break
                except ValueError:
                    continue
        self.points=copy.deepcopy(self.previous.get("image_points",[]));self.world=copy.deepcopy(self.previous.get("world_points",[]))
        self.controls=list(zip(copy.deepcopy(self.previous.get("control_image_points",[])),copy.deepcopy(self.previous.get("control_world_points",[]))))
        self.regions={k:copy.deepcopy(self.previous[k]) for k in ("visible_image","people_roi","ball_roi","obstacles","net","net_region","referee_region") if k in self.previous}
        self.line_constraints=copy.deepcopy(self.previous.get('line_constraints',[]))
        self.zones=copy.deepcopy(self.previous.get("zones",[]));self.lens_model=copy.deepcopy(self.previous.get("lens_model"))
        self.current=[];self.result=copy.deepcopy(self.previous) if self.previous.get("matrix") is not None else None
        layout=W.QVBoxLayout(self);self.canvas=Canvas();self.canvas.clicked.connect(self.click);layout.addWidget(self.canvas,1)
        row=W.QHBoxLayout();layout.addLayout(row)
        self.mode=combo([("Опорные точки пола","fit"),("Независимые контрольные точки","control"),
            ("Видимый пол","visible_image"),("Поиск людей","people_roi"),("Поиск мяча, включая воздух","ball_roi"),
            ("Постоянная преграда","obstacle"),("Отрезок сетки, не пол","net"),("Приблизительная зона 1–6","zone"),
            ('Подписанный отрезок линии пола','floor_line'),('Область полотна сетки','net_region'),('Область судьи — подсказка','referee_region')])
        row.addWidget(self.mode);self.landmark=combo([(k,v) for k,v in LANDMARKS.items()]);row.addWidget(self.landmark)
        button("Завершить область",self.finish_polygon,row);button("Отменить точку",self.undo_point,row)
        from .line_geometry import LINE_LABELS
        line_row=W.QHBoxLayout();layout.addLayout(line_row);line_row.addWidget(W.QLabel('Назначение линии пола'))
        self.floor_line=combo([(label,key) for key,label in LINE_LABELS.items()]);line_row.addWidget(self.floor_line)
        button('Удалить последний отрезок',self.remove_line,line_row)
        settings=W.QHBoxLayout();layout.addLayout(settings)
        settings.addWidget(W.QLabel("Зона"));self.zone_number=W.QSpinBox();self.zone_number.setRange(1,6);settings.addWidget(self.zone_number)
        self.zone_team=combo([("Команда не установлена",None)]+[(t["name"],t["id"]) for t in project.all("team")]);settings.addWidget(self.zone_team)
        self.zone_start=W.QDoubleSpinBox();self.zone_end=W.QDoubleSpinBox()
        for widget,value in ((self.zone_start,0),(self.zone_end,source["duration"])):
            widget.setRange(0,source["duration"]);widget.setDecimals(3);widget.setValue(value)
        settings.addWidget(W.QLabel("Действует с"));settings.addWidget(self.zone_start);settings.addWidget(W.QLabel("до, с"));settings.addWidget(self.zone_end)
        button("Удалить последнюю зону",self.remove_zone,settings)
        scope=W.QHBoxLayout();layout.addLayout(scope);scope.addWidget(W.QLabel("Ракурс действует с"))
        self.profile_start=W.QDoubleSpinBox();self.profile_end=W.QDoubleSpinBox()
        default_start=self.previous.get("start",timestamp if prior else 0)
        default_end=self.previous.get("end")
        if default_end is None:default_end=min([p["start"] for p in prior if p.get("start",0)>timestamp]+[source["duration"]])
        for widget,value in ((self.profile_start,default_start),(self.profile_end,default_end)):
            widget.setRange(0,source["duration"]);widget.setDecimals(3);widget.setValue(value)
        scope.addWidget(self.profile_start);scope.addWidget(W.QLabel("до, с"));scope.addWidget(self.profile_end)
        lens_row=W.QHBoxLayout();layout.addLayout(lens_row)
        self.lens_status=W.QLabel();lens_row.addWidget(self.lens_status,1)
        button("Загрузить измеренную калибровку объектива",self.load_lens,lens_row)
        button("Без коррекции объектива",self.clear_lens,lens_row)
        self.list=W.QPlainTextEdit();self.list.setReadOnly(True);self.list.setMaximumHeight(100);layout.addWidget(self.list)
        self.status=W.QLabel("Мало ориентиров — сохраните зоны или координаты видео. Сетка и головы не являются точками пола.")
        self.status.setWordWrap(True);layout.addWidget(self.status)
        row=W.QHBoxLayout();layout.addLayout(row)
        button("Предложить видимые линии",self.lines,row);button("Уточнить по линиям пола",self.refine_lines,row)
        button("Очистить опорные точки",self.clear,row)
        button("Проверить совмещение и доступную аналитику",self.check,row);button("Сохранить профиль",self.save,row)
        self.redraw()

    def profile_base(self):
        result=copy.deepcopy(self.result or {"geometry":"video","verified":False,"matrix":None,"coordinate_space":"rotated_original"})
        result.update(source_id=self.source["id"],source_sha256=self.source["sha256"],
            image_size=[self.rgb.shape[1],self.rgb.shape[0]],lens_model=copy.deepcopy(self.lens_model),zones=copy.deepcopy(self.zones),
            start=self.profile_start.value(),end=self.profile_end.value(),
            image_points=copy.deepcopy(self.points),world_points=copy.deepcopy(self.world),
            line_constraints=copy.deepcopy(self.line_constraints),
            control_image_points=[p[0] for p in self.controls],control_world_points=[p[1] for p in self.controls],**self.regions)
        return result

    def redraw(self):
        shapes=[{"polygon":z["polygon"],"label":f"≈ зона {z['zone']} · {z.get('team_id') or '?'}"} for z in self.zones]
        for key,label in (("visible_image","Видимый пол"),("people_roi","Поиск людей"),("ball_roi","Поиск мяча"),
                          ('net','Сетка — не пол'),('net_region','Полотно сетки'),('referee_region','Судья — только подсказка')):
            if self.regions.get(key):shapes.append({"polygon":self.regions[key],"label":label})
        shapes += [{"polygon":p,"label":"Преграда","color":"#ff8866"} for p in self.regions.get("obstacles",[])]
        shapes += [{'polygon':s['image'],'label':s['line'],'color':'#ffe568'} for s in self.line_constraints]
        if self.current:shapes.append({"polygon":self.current,"label":"Не завершено"})
        self.canvas.show_frame(self.rgb,points=self.points+[p[0] for p in self.controls],
            geometry_profile=self.profile_base() if self.result else None,polygons=shapes)
        self.list.setPlainText(json.dumps({"ориентиры":list(zip(self.points,self.world)),"контроли":self.controls,'отрезки':self.line_constraints,
            "зоны":[{"зона":z["zone"],"команда":z.get("team_id"),"с":z.get("start"),"до":z.get("end")} for z in self.zones]},ensure_ascii=False))
        self.lens_status.setText("Измеренная модель: "+self.lens_model["type"] if self.lens_model else "Дисторсия не измерена; 0.5× автоматически не считается fisheye")

    def click(self,x,y):
        if not (0<=x<self.rgb.shape[1] and 0<=y<self.rgb.shape[0]):return
        mode=self.mode.currentData()
        if mode=="fit":self.points.append([x,y]);self.world.append(self.landmark.currentData());self.result=None
        elif mode=="control":self.controls.append(([x,y],self.landmark.currentData()));self.result=None
        else:self.current.append([x,y])
        self.redraw()

    def finish_polygon(self):
        from .calibration import validate_profile
        mode=self.mode.currentData()
        if mode in ("fit","control"):return
        if len(self.current)<(2 if mode in ('net','floor_line') else 3):self.status.setText("Недостаточно точек: для области нужны минимум три, для отрезка — две");return
        backup=copy.deepcopy((self.regions,self.zones,self.line_constraints,self.result))
        try:
            if mode=='floor_line':
                if len(self.current)!=2:raise ValueError('Отрезок задаётся ровно двумя точками')
                self.line_constraints.append({'line':self.floor_line.currentData(),'image':copy.deepcopy(self.current)});self.result=None
            elif mode=="zone":
                if not self.zone_team.currentData():raise ValueError("Выберите команду для зоны 1–6")
                self.zones.append({"zone":self.zone_number.value(),"polygon":copy.deepcopy(self.current),
                    "team_id":self.zone_team.currentData(),"source_id":self.source["id"],"start":self.zone_start.value(),"end":self.zone_end.value()})
            elif mode=="obstacle":self.regions.setdefault("obstacles",[]).append(copy.deepcopy(self.current))
            else:self.regions[mode]=copy.deepcopy(self.current)
            validate_profile(self.profile_base(),source_id=self.source["id"],source_sha256=self.source["sha256"],image_size=[self.rgb.shape[1],self.rgb.shape[0]])
            self.current=[];self.redraw()
        except ValueError as exc:self.regions,self.zones,self.line_constraints,self.result=backup;self.status.setText(str(exc))

    def undo_point(self):
        mode=self.mode.currentData()
        if mode=="fit" and self.points:self.points.pop();self.world.pop();self.result=None
        elif mode=="control" and self.controls:self.controls.pop();self.result=None
        elif self.current:self.current.pop()
        self.redraw()

    def remove_zone(self):
        if self.zones:self.zones.pop();self.redraw()

    def remove_line(self):
        if self.line_constraints:self.line_constraints.pop();self.result=None;self.redraw()

    def clear(self):
        self.points=[];self.world=[];self.controls=[];self.line_constraints=[];self.current=[];self.result=None;self.redraw()

    def load_lens(self):
        path,_=W.QFileDialog.getOpenFileName(self,"Измеренная модель объектива: pinhole/fisheye, K, D",filter="JSON (*.json)")
        if not path:return
        previous=self.lens_model;previous_result=self.result
        try:
            from .calibration import validate_profile
            data=json.loads(Path(path).read_text(encoding="utf-8"));self.lens_model=data.get("lens_model",data)
            self.result=None
            validate_profile(self.profile_base(),source_id=self.source["id"],source_sha256=self.source["sha256"],image_size=[self.rgb.shape[1],self.rgb.shape[0]])
            self.result=None;self.redraw();self.status.setText("Калибровка объектива загружена. Повторите проверку опорных и независимых точек")
        except (ValueError,KeyError,TypeError) as exc:self.lens_model=previous;self.result=previous_result;self.status.setText(str(exc))

    def clear_lens(self):self.lens_model=None;self.result=None;self.redraw()

    def lines(self):
        hsv=cv2.cvtColor(self.rgb,cv2.COLOR_RGB2HSV);mask=cv2.inRange(hsv,np.array([0,0,150]),np.array([179,65,255]))
        lines=cv2.HoughLinesP(cv2.Canny(mask,60,140),1,np.pi/180,100,minLineLength=max(40,self.rgb.shape[1]//12),maxLineGap=20)
        image=self.rgb.copy()
        for line in [] if lines is None else lines[:35]:
            x1,y1,x2,y2=map(int,line[0]);cv2.line(image,(x1,y1),(x2,y2),(40,240,200),2)
        self.canvas.show_frame(image,points=self.points)
        self.status.setText("Отрезки могут относиться к окнам и стенам. Выберите известные пересечения самостоятельно; это не проверенное автоматическое совмещение")

    def check(self):
        from .calibration import validate_profile
        try:
            if self.points and self.result is None:
                self.result=fit(self.points,self.world,[p[0] for p in self.controls],[p[1] for p in self.controls],lens_model=self.lens_model)
            elif self.line_constraints and self.result is None:
                from .line_geometry import fit_from_lines
                self.result=fit_from_lines(self.line_constraints,[self.rgb.shape[1],self.rgb.shape[0]],
                    [p[0] for p in self.controls],[p[1] for p in self.controls],self.lens_model)
            profile=validate_profile(self.profile_base(),source_id=self.source["id"],source_sha256=self.source["sha256"],image_size=[self.rgb.shape[1],self.rgb.shape[0]])
            self.redraw()
            if profile.get("geometry")=="court":summary="Доступны координаты пола в метрах; мяч в воздухе на пол не переносится"
            elif self.zones:summary=f"Доступны приблизительные зоны ({len(self.zones)}); точные метры недоступны"
            else:summary="Доступны координаты видео; точные метры и подтверждённые зоны недоступны"
            if self.result and self.result.get('fit_rmse_m') is not None:
                summary+=f". RMS: {self.result['fit_rmse_m']:.3f} м; контроль: {self.result.get('control_rmse_m')}; чувствительность: {self.result.get('click_sensitivity_2px_p95_m')} м"
            self.status.setText(summary)
        except ValueError as exc:self.result=None;self.redraw();self.status.setText(str(exc)+". Можно сохранить ограниченный режим")

    def refine_lines(self):
        from .line_refinement import refine_floor_lines
        if self.result is None:self.check()
        if self.result is None:self.status.setText("Сначала задайте минимум четыре известных невырожденных ориентира пола");return
        try:
            candidate,diagnostics=refine_floor_lines(self.profile_base(),self.rgb,self.source["id"])
            self.result=candidate;self.redraw()
            self.status.setText("Уточнение предложено; независимые контроли сохранены. " if diagnostics.get("applied") else "Уточнение не применено. ")
            self.status.setText(self.status.text()+str(diagnostics.get("reason",""))+" Геометрия: "+candidate["geometry"])
        except ValueError as exc:self.status.setText(str(exc))

    def save(self):
        from .calibration import map_floor,validate_profile
        if self.current:self.status.setText("Сначала завершите текущую область или отмените её точки");return
        self.check()
        try:
            result=validate_profile(self.profile_base(),source_id=self.source["id"],source_sha256=self.source["sha256"],image_size=[self.rgb.shape[1],self.rgb.shape[0]])
            if not result['start']<=self.timestamp<=result['end']:
                raise ValueError("Интервал ракурса должен включать время показанного опорного кадра")
            result.update(id=uid(),parent_id=self.previous.get("id"),version=self.previous.get("version",0)+1,
                reference_time=self.timestamp)
            if result.get("verified") and result.get("visible_image"):
                visible=[map_floor(result,p,source_id=self.source["id"],time=self.timestamp,on_floor=True)["floor_point"] for p in result["visible_image"]]
                result["visible_floor"]=visible if all(point is not None for point in visible) else []
            ref=self.project.root/"cache"/(result["id"]+"-reference.jpg")
            write_image(ref,cv2.cvtColor(self.rgb,cv2.COLOR_RGB2BGR));result["reference_image"]=self.project.relpath(ref)
            result["distortion_status"]="измеренная модель загружена" if self.lens_model else "не измерена; fisheye не предполагается"
            self.project.put("calibration",result);self.accept()
        except (ValueError,KeyError) as exc:self.status.setText(str(exc))


class MainWindow(W.QMainWindow):
    def __init__(self,root):
        super().__init__(); self.project=Project(root);self.worker=None;self.source=None;self.current=None;self.iterator=None
        self.playing=False;self.speed=1;self.timer=C.QTimer(self);self.timer.timeout.connect(self.tick)
        self.setWindowTitle("VolleyVision • локальная аналитика 6×6");self.resize(1450,960)
        container=W.QWidget();self.setCentralWidget(container);layout=W.QVBoxLayout(container)
        banner=W.QLabel("Начальная настройка → анализ → автоматический отчёт. Распознавание волейбольных действий пока ограничено кандидатами касания; точность не измерена.")
        banner.setStyleSheet("padding:12px;background:#ffe6b0;color:#4a3518");banner.setWordWrap(True);layout.addWidget(banner)
        top=W.QHBoxLayout();layout.addLayout(top)
        button("Создать/открыть",self.open_project,top);button("Импорт видео",self.import_files,top)
        self.sources=W.QComboBox();self.sources.currentIndexChanged.connect(self.choose_source);top.addWidget(self.sources,1)
        button("Восстановить путь",self.relink,top);button("Лёгкая копия",self.proxy,top)
        split=W.QSplitter();layout.addWidget(split,1)
        left=W.QWidget();left_layout=W.QVBoxLayout(left);self.canvas=Canvas();left_layout.addWidget(self.canvas,1);split.addWidget(left)
        controls=W.QHBoxLayout();left_layout.addLayout(controls)
        button("◀ кадр",lambda:self.step(-1),controls);button("▶ / пауза",self.toggle,controls);button("кадр ▶",lambda:self.step(1),controls)
        self.time=W.QDoubleSpinBox();self.time.setDecimals(6);self.time.setRange(0,1e8);self.time.editingFinished.connect(lambda:self.seek(self.time.value()));controls.addWidget(self.time)
        self.rate=combo([("0.25×",.25),("0.5×",.5),("1×",1.0),("2×",2.0)],1.0);controls.addWidget(self.rate)
        self.slider=W.QSlider(C.Qt.Orientation.Horizontal);self.slider.setRange(0,10000);self.slider.sliderReleased.connect(lambda:self.seek(self.slider.value()/10000*self.source["duration"]) if self.source else None);left_layout.addWidget(self.slider)
        self.tabs=W.QTabWidget();split.addWidget(self.tabs)
        setup=W.QWidget();s=W.QVBoxLayout(setup);self.tabs.addTab(setup,"Настройка")
        self.info=W.QPlainTextEdit();self.info.setReadOnly(True);s.addWidget(self.info)
        button("Команды и игроки",self.roster,s);button("Площадка и области наблюдения",self.calibrate,s)
        s.addWidget(W.QLabel('Как сопоставлять игроков'))
        self.identification=combo([(label,key) for key,label in IDENTITY_MODES.items()],identity_mode(self.project));s.addWidget(self.identification)
        self.identification.currentIndexChanged.connect(self.identity_changed)
        button('Наша команда, образцы формы и либеро',self.kits,s)
        button('Ручной счёт и партии (необязательно)',self.scores,s)
        button('Проверить встроенные модели' if os.environ.get('VOLLEYVISION_BUNDLE_ROOT') else "Подготовить модели (нужна сеть)",self.fetch_models,s)
        if os.environ.get('VOLLEYVISION_BUNDLE_ROOT'):
            button('Ускорение NVIDIA: подготовить автоматически',self.prepare_nvidia,s)
        self.setup_scan=button("Предварительный проход людей и номеров",lambda:self.analysis(True),s)
        button("Галерея: подтвердить игроков",self.gallery,s)
        button('Обновить сопоставление игроков без повторной детекции',self.refresh_identities,s)
        button("Подключить обученную модель действий",self.temporal_model,s)
        button("Стороны команд и интервалы состава",self.context,s)
        button("Запустить анализ → отчёт",lambda:self.analysis(False),s)
        self.auto_status=W.QLabel("Непривязанные игроки не блокируют запуск. После старта вопросов нет. Отмена сохраняет обработанные интервалы.")
        self.auto_status.setWordWrap(True);s.addWidget(self.auto_status)
        ev=W.QWidget();e=W.QVBoxLayout(ev);self.tabs.addTab(ev,"Эпизоды (необязательно)")
        self.only_unknown=W.QCheckBox("Только непроверенные");self.only_unknown.toggled.connect(self.refresh_events);e.addWidget(self.only_unknown)
        self.event_filter=W.QLineEdit();self.event_filter.setPlaceholderText("Поиск: игрок, действие, команда, исход…");self.event_filter.textChanged.connect(self.refresh_events);e.addWidget(self.event_filter)
        self.events=W.QTableWidget(0,4);self.events.setHorizontalHeaderLabels(["Время","Действие","Игрок","Проверка"])
        self.events.setSelectionBehavior(W.QAbstractItemView.SelectionBehavior.SelectRows);self.events.setEditTriggers(W.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.events.cellDoubleClicked.connect(self.edit_selected);e.addWidget(self.events)
        row=W.QHBoxLayout();e.addLayout(row);button("Новое (N)",self.new_event,row);button("Изменить",self.edit_selected,row);button("Отменить правку",self.undo,row)
        button("Розыгрыш и результат",self.rally,e)
        stats=W.QWidget();st=W.QVBoxLayout(stats);self.tabs.addTab(stats,"Отчёт")
        self.mode=combo([(v,k) for k,v in MODES.items()],"automatic");st.addWidget(self.mode)
        self.filters=W.QLineEdit();self.filters.setPlaceholderText('Фильтры JSON, например {"team_id":"A","type":"attack"}');st.addWidget(self.filters)
        button("Пересчитать и показать",self.show_stats,st);button("Экспорт CSV/XLSX/JSON/HTML",self.export,st)
        button("Экспорт с MP4-клипами",lambda:self.export(True),st)
        clip_settings=W.QHBoxLayout();st.addLayout(clip_settings)
        self.before=W.QDoubleSpinBox();self.before.setRange(0,60);self.before.setValue(2)
        self.after=W.QDoubleSpinBox();self.after.setRange(0,60);self.after.setValue(3)
        clip_settings.addWidget(W.QLabel("До события, с"));clip_settings.addWidget(self.before)
        clip_settings.addWidget(W.QLabel("После, с"));clip_settings.addWidget(self.after)
        self.stats=W.QTextBrowser();st.addWidget(self.stats)
        bottom=W.QHBoxLayout();layout.addLayout(bottom);self.progress=W.QProgressBar();bottom.addWidget(self.progress)
        self.cancel_button=button("Отмена",lambda:self.worker.stop.set() if self.worker else None,bottom)
        self.message=W.QLabel("Готово");layout.addWidget(self.message)
        for key,fn in (("Space",self.toggle),("Right",lambda:self.step(1)),("Left",lambda:self.step(-1)),("N",self.new_event),("Ctrl+Z",self.undo)):
            shortcut=G.QShortcut(G.QKeySequence(key),self);shortcut.activated.connect(fn)
        self.refresh()

    def error(self,exc):W.QMessageBox.warning(self,"VolleyVision",str(exc))

    def run_job(self,fn):
        if self.worker: self.message.setText("Дождитесь текущей операции или отмените её");return
        self.playing=False;self.timer.stop();self.worker=Worker(self.project.root,fn)
        self.worker.signals.progress.connect(lambda v,text:(self.progress.setValue(int(v*100)),self.message.setText(text)))
        self.worker.signals.finished.connect(self.done);self.worker.signals.failed.connect(self.failed)
        self.message.setText("Выполняется…");C.QThreadPool.globalInstance().start(self.worker)
        self.identification.setEnabled(False)

    def done(self,result):
        self.worker=None;self.progress.setValue(100);self.message.setText(result[-500:]);self.refresh()

    def failed(self,text):self.worker=None;self.error(text);self.refresh()

    def refresh(self):
        self.identification.blockSignals(True);self.identification.setCurrentIndex(self.identification.findData(identity_mode(self.project)));self.identification.blockSignals(False)
        self.identification.setEnabled(self.worker is None)
        self.setup_scan.setText('Предварительный проход людей и внешности' if identity_mode(self.project)=='appearance' else 'Предварительный проход людей и номеров')
        current=self.source["id"] if self.source else None;self.sources.blockSignals(True);self.sources.clear()
        for s in self.project.all("source"):self.sources.addItem(Path(s["path"]).name,s["id"])
        index=self.sources.findData(current);self.sources.setCurrentIndex(max(0,index));self.sources.blockSignals(False)
        if self.sources.count():self.choose_source()
        self.refresh_events();self.show_stats()

    def identity_changed(self):
        set_identity_mode(self.project,self.identification.currentData())
        self.setup_scan.setText('Предварительный проход людей и внешности' if identity_mode(self.project)=='appearance' else 'Предварительный проход людей и номеров')
        self.message.setText('Выбран режим: '+IDENTITY_MODES[identity_mode(self.project)]+'. Новые связи пока остаются предположениями.')

    def choose_source(self):
        self.source=self.project.get("source",self.sources.currentData())
        if self.source:
            self.info.setPlainText(json.dumps({k:self.source.get(k) for k in ("codec","width","height","display_width","display_height","average_rate","variable_fps","duration","rotation","hdr","gap_unknown","proxy_verified")},ensure_ascii=False,indent=2))
            self.seek(0)

    def seek(self,t):
        if not self.source:return
        self.playing=False;self.timer.stop();self.iterator=None
        try:
            use_proxy=self.source.get("proxy_verified",False)
            path=self.project.resolve(self.source,proxy=use_proxy);meta=self.source["proxy_metadata"] if use_proxy else self.source
            row=self.project.db.execute("SELECT MAX(time) FROM frames WHERE source_id=?",(self.source["id"],)).fetchone()
            last_time=row[0] if row and row[0] is not None else max(0,self.source["duration"]-.001)
            self.iterator=frames(path,meta,start=min(max(0,t),last_time))
            self.display(next(self.iterator),use_proxy)
        except (ValueError,FileNotFoundError,StopIteration) as exc:self.error(exc)

    def display(self,frame,use_proxy=False):
        self.current=frame;self.time.blockSignals(True);self.time.setValue(frame.time);self.time.blockSignals(False)
        self.slider.setValue(int(frame.time/max(self.source["duration"],.01)*10000))
        overlays=[];jobs=[j for j in self.project.all("job") if j["source_id"]==self.source["id"]]
        if jobs:
            job=jobs[-1];row=self.project.db.execute("SELECT body FROM observations WHERE run_id=? AND time<=? ORDER BY time DESC LIMIT 1",(job["id"],frame.time+.04)).fetchone()
            if row:
                obs=json.loads(row[0])
                if abs(obs["time"]-frame.time)<.22:
                    overlays=obs["people"]+obs["balls"]
                    scale=frame.rgb.shape[1]/self.source["display_width"]
                    for d in overlays:
                        d["bbox"]=[x*scale for x in d["bbox"]]
                        b=self.project.track_identity(job["id"],d["track_id"],obs["time"])
                        if d.get('scene_role')=='referee':d['label']='Судья'
                        elif b and b.get('player_id'):
                            card=self.project.get('player',b['player_id']);d['label']=player_label(card) if card else b['player_id']
                        else:
                            proposal=d.get('team_proposal',{});team_id=(b or {}).get('team_id') or proposal.get('team_id')
                            team=self.project.get('team',team_id) if team_id else None
                            prefix=team['name']+('?' if not b and proposal.get('evidence')!='confirmed' else '')+' · ' if team else ''
                            d['label']=prefix+d['track_id'][-12:]
                        d["label"]+=f" Δt={obs['time']-frame.time:+.2f}s"
        self.canvas.show_frame(frame.rgb,overlays)

    def toggle(self):
        if not self.source or not self.current:return
        if self.playing:self.playing=False;self.timer.stop();return
        self.playing=True;self.speed=self.rate.currentData();self.play_start=time.monotonic();self.media_start=self.current.time;self.timer.start(1)

    def tick(self):
        if not self.playing:return
        try:
            frame=next(self.iterator);self.display(frame,bool(self.source.get("proxy_verified")))
            wait=(frame.time-self.media_start)/self.speed-(time.monotonic()-self.play_start)
            self.timer.setInterval(max(1,round(wait*1000)))
        except StopIteration:self.playing=False;self.timer.stop()
        except Exception as exc:self.playing=False;self.timer.stop();self.error(exc)

    def step(self,direction):
        if not self.current:return
        op=">" if direction>0 else "<";order="ASC" if direction>0 else "DESC"
        row=self.project.db.execute(f"SELECT time FROM frames WHERE source_id=? AND time {op} ? ORDER BY time {order} LIMIT 1",(self.source["id"],self.current.time+direction*1e-5)).fetchone()
        if row:self.seek(row[0])

    def open_project(self):
        if self.worker:return
        path=W.QFileDialog.getExistingDirectory(self,"Открыть проект или создать в пустой папке")
        if not path:return
        try:
            new=Project(path,create=True);self.project.close();self.project=new;self.source=None;self.current=None;self.refresh()
        except Exception as exc:self.error(exc)

    def import_files(self):
        paths,_=W.QFileDialog.getOpenFileNames(self,"Оригинальные последовательные части",filter="Видео (*.MOV *.mov *.MP4 *.mp4);;Все файлы (*)")
        if paths:
            def task(p,progress,cancel):
                for path in paths:import_video(p,path,progress=progress,cancel=cancel)
                return "Импорт завершён. Неизвестные разрывы между частями отмечены отдельно"
            self.run_job(task)

    def relink(self):
        if not self.source:return
        path,_=W.QFileDialog.getOpenFileName(self,"Новый путь к тому же исходнику")
        if path:
            sid=self.source["id"]
            def task(p,progress,cancel):
                from .media import sha256
                s=p.get("source",sid)
                if sha256(path,cancel)!=s["sha256"]:raise ValueError("Это другой файл; контрольная сумма не совпала")
                s["path"]=p.relpath(path);p.put("source",s);return "Путь восстановлен"
            self.run_job(task)

    def proxy(self):
        if self.source:
            sid=self.source["id"];self.run_job(lambda p,progress,cancel:create_proxy(p,p.get("source",sid),progress=progress,cancel=cancel))

    def roster(self):
        from .setup_ui import RosterDialog
        RosterDialog(self.project,self).exec();self.refresh_events();self.show_stats()

    def calibrate(self):
        if not self.source or not self.current:return
        try:
            # Calibration always uses upright original pixel coordinates.
            rgb=frame_at(self.project.resolve(self.source),self.source,self.current.time).rgb
            CalibrationDialog(self.project,self.source,rgb,self,timestamp=self.current.time).exec()
        except Exception as exc:self.error(exc)

    def kits(self):
        from .setup_ui import KitDialog
        try:
            frame=frame_at(self.project.resolve(self.source),self.source,self.current.time) if self.source and self.current else None
            KitDialog(self.project,self.source,frame.rgb if frame else None,frame.time if frame else 0,self).exec()
            from .identity import second_pass
            for job in self.project.all('job'):second_pass(self.project,job['id'])
            self.show_stats()
        except (ValueError,RuntimeError,FileNotFoundError) as exc:self.error(exc)

    def scores(self):
        from .setup_ui import ScoreDialog
        ScoreDialog(self.project,self.source['id'] if self.source else None,self.current.time if self.current else 0,self).exec()
        self.show_stats()

    def fetch_models(self):
        def task(p,progress,cancel):
            from .models import fetch
            include_ocr=identity_mode(p)!='appearance'
            fetch(p.root/"models",include_ocr=include_ocr);return 'Локальные модели подготовлены'+(' без OCR: номера не требуются' if not include_ocr else ', включая OCR')
        self.run_job(task)

    def analysis(self,setup_only):
        sources=self.project.all("source")
        if not sources:return
        def task(p,progress,cancel):
            from .vision import analyze
            settings=p.get("analysis_config","default") or {}
            configuration={"temporal_artifact":settings["temporal_artifact"]} if settings.get("temporal_artifact") else {}
            for s in sources:analyze(p,s["id"],configuration,progress=progress,cancel=cancel,setup_only=setup_only)
            return "Предварительный проход завершён; галерея доступна" if setup_only else "Анализ завершён; отчёт в exports/automatic/report.html. Неопределённости не требуют ответа"
        self.run_job(task)

    def prepare_nvidia(self):
        from .portable_gpu import prepare
        bundle=os.environ.get('VOLLEYVISION_BUNDLE_ROOT')
        if bundle:self.run_job(lambda p,progress,cancel:prepare(bundle,progress,cancel))

    def temporal_model(self):
        path,_=W.QFileDialog.getOpenFileName(self,"Manifest обученной временной модели (готовые веса волейбола не входят в проект)",filter="JSON (*.json)")
        if not path:return
        try:
            from .temporal import TemporalPredictor
            predictor=TemporalPredictor(path)
            self.project.put("analysis_config",{"id":"default","temporal_artifact":self.project.relpath(path)})
            self.message.setText("Модель действий подключена. "+("Политика автоматического принятия включена; поля участника и исхода независимы" if predictor.policy.get("enabled") else "Результаты останутся гипотезами до оценки на реальном эталоне"))
        except (ValueError,RuntimeError,ImportError,FileNotFoundError) as exc:self.error(exc)

    def gallery(self):
        from .gallery import GalleryDialog
        dialog=GalleryDialog(self.project,self)
        def preview(source_id,t):
            index=self.sources.findData(source_id)
            if index>=0:
                self.sources.setCurrentIndex(index);self.seek(t)
        dialog.previewRequested.connect(preview)
        dialog.bindingsChanged.connect(self.show_stats)
        dialog.exec()
        self.show_stats()
        if self.project.all('job') and not self.worker:self.refresh_identities()

    def refresh_identities(self):
        def task(p,progress,cancel):
            from .identity import second_pass
            latest={}
            for job in p.all('job'):latest[job['source_id']]=job
            for job in latest.values():second_pass(p,job['id'],read_numbers=True,cancel=cancel,progress=progress)
            export_all(p)
            return 'Сопоставление обновлено; новые связи остаются гипотезами. Детекция не повторялась'
        self.run_job(task)

    def context(self):
        d=W.QDialog(self);d.setWindowTitle("Интервалы: стороны / состав / замена / расстановка");d.resize(800,600);layout=W.QVBoxLayout(d)
        kind=combo([("Стороны команд","side"),("Составы (неизвестные интервалы не заполняются)","lineup"),("Партии","set")]);layout.addWidget(kind)
        editor=W.QPlainTextEdit();layout.addWidget(editor)
        def load():editor.setPlainText(json.dumps(self.project.all(kind.currentData()),ensure_ascii=False,indent=2))
        kind.currentIndexChanged.connect(load);load()
        layout.addWidget(W.QLabel("side: id, team_id, source_id, start, end, side=left/right. lineup: id, team_id, source_id, start, end, players, rotation. Все интервалы в секундах исходника."))
        def save():
            try:
                values=json.loads(editor.toPlainText())
                for obj in values:self.project.put(kind.currentData(),obj)
                d.accept()
            except (ValueError,TypeError) as exc:self.error(exc)
        button("Сохранить",save,layout);d.exec()

    def refresh_events(self):
        rows=self.project.events();text=self.event_filter.text().casefold()
        rows=[e for e in rows if (not self.only_unknown.isChecked() or e["review"]=="unreviewed") and (not text or text in json.dumps(e,ensure_ascii=False).casefold() or text in EVENT_TYPES[e["type"]].casefold())]
        self.events.setRowCount(len(rows))
        for i,e in enumerate(rows):
            for j,value in enumerate((f"{e['start']:.3f}",EVENT_TYPES[e["type"]],e.get("player_id") or "неизвестен",e["review"])):
                item=W.QTableWidgetItem(value);item.setData(C.Qt.ItemDataRole.UserRole,e);self.events.setItem(i,j,item)
        self.events.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch)

    def edit_selected(self,*args):
        row=self.events.currentRow()
        if row<0:return
        e=self.events.item(row,0).data(C.Qt.ItemDataRole.UserRole)
        if self.source and self.source["id"]!=e["source_id"]:self.sources.setCurrentIndex(self.sources.findData(e["source_id"]))
        self.seek(e["start"]);self.edit(e)

    def new_event(self):
        if self.source and self.current:self.edit(event(self.source["id"],self.current.time,"attack"))

    def edit(self,e):
        d=EventDialog(self.project,e,self)
        if d.exec()==W.QDialog.DialogCode.Accepted:
            try:self.project.save_event(d.value,e.get("revision"));self.refresh_events();self.show_stats()
            except ValueError as exc:self.error(exc)

    def undo(self):self.project.undo();self.refresh_events();self.show_stats()

    def rally(self):
        if not self.source:return
        d=W.QDialog(self);d.setWindowTitle("Результат розыгрыша (необязательно)");form=W.QFormLayout(d)
        teams=[(t["name"],t["id"]) for t in self.project.all("team")]
        server=combo([("Неизвестна",None)]+teams);winner=combo([("Неизвестен",None)]+teams)
        start=W.QDoubleSpinBox();end=W.QDoubleSpinBox()
        party=combo([('Партия не указана',None)]+[(f"Партия {s.get('number','?')}",s['id']) for s in self.project.all('set')])
        existing=combo([('Новый розыгрыш',None)]+[(f"{r.get('start',0):.3f}–{r.get('end',0):.3f} с",r['id'])
            for r in self.project.all('rally') if r.get('source_id')==self.source['id']])
        form.addRow('Создать / исправить',existing)
        def load_rally():
            r=self.project.get('rally',existing.currentData())
            if r:
                start.setValue(r['start']);end.setValue(r['end']);server.setCurrentIndex(server.findData(r.get('server_team_id')))
                winner.setCurrentIndex(winner.findData(r.get('winner_team_id')));party.setCurrentIndex(party.findData(r.get('set_id')))
        existing.currentIndexChanged.connect(load_rally)
        for box in (start,end):box.setRange(0,self.source["duration"]);box.setDecimals(3);box.setValue(self.time.value())
        form.addRow("Начало",start);form.addRow("Конец",end);form.addRow("Подавала",server);form.addRow("Победитель",winner)
        form.addRow('Партия',party)
        def save():
            if end.value()<start.value():self.error("Конец раньше начала");return
            old=self.project.get('rally',existing.currentData())
            value={"id":existing.currentData() or uid(),"source_id":self.source["id"],"start":start.value(),"end":end.value(),
                'set_id':party.currentData(),'origin':'manual',
                "server_team_id":server.currentData(),"winner_team_id":winner.currentData(),
                "evidence":{"server_team_id":"confirmed" if server.currentData() else "unknown","winner_team_id":"confirmed" if winner.currentData() else "unknown"}}
            self.project.put('rally_history',{'id':uid(),'before':old,'after':value});self.project.put('rally',value)
            d.accept();self.show_stats()
        b=W.QPushButton("Сохранить известные поля");b.clicked.connect(save);form.addRow(b);d.exec()

    def show_stats(self):
        try:
            filters=json.loads(self.filters.text()) if self.filters.text().strip() else {}
            data=report(self.project,self.mode.currentData(),filters)
            text=[data["mode_label"],f"Записано событий: {len(data['events'])}; вне основных таблиц: {data['excluded_events']}; полнота не определена"]
            team_names={t['id']:t['name'] for t in data['teams']}
            for s in data['manual_score']['sets']:
                if s['final_scores']:
                    text.append(f"Партия {s['number']}, ручной итог: "+' · '.join(f"{team_names.get(t,t)} {v}" for t,v in s['final_scores'].items()))
            if data['manual_score']['conflicts']:text.append(f"Противоречий ручного счёта: {len(data['manual_score']['conflicts'])}; подробности в отчёте.")
            for p in data["players"]:
                m=p["attack_efficiency"];text.append(f"{p['name']} [{p['team_id']}]: атаки {p['attacks_recorded']}; эффективность {m['value'] if m['value'] is not None else 'нет данных'} ({m['numerator']}/{m['denominator']}), неизвестных исходов {m['missing']}")
            self.stats.setPlainText("\n\n".join(text))
        except ValueError as exc:self.message.setText(str(exc))

    def export(self,clips=False):
        try:filters=json.loads(self.filters.text()) if self.filters.text().strip() else {}
        except ValueError as exc:self.error(exc);return
        mode=self.mode.currentData()
        before,after=self.before.value(),self.after.value()
        self.run_job(lambda p,progress,cancel:export_all(p,mode=mode,filters=filters,clips=bool(clips),before=before,after=after,cancel=cancel))

    def closeEvent(self,e):
        if self.worker:
            self.worker.stop.set();self.message.setText("Сохраняется контрольная точка. Закройте окно после завершения операции");e.ignore();return
        self.timer.stop();self.project.close();e.accept()


def run(root):
    app=W.QApplication.instance() or W.QApplication([])
    app.setApplicationName("VolleyVision");app.setStyle("Fusion")
    from .simple_ui import SimpleWindow
    window=SimpleWindow(root);window.show()
    return app.exec()
