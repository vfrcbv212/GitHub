"""Optional direct scene annotation for a stationary camera."""
from __future__ import annotations

import copy
import json
import math

import cv2
import numpy as np
from PySide6 import QtCore as C,QtGui as G,QtWidgets as W

from .gui import Canvas,Worker,button
from .media import frame_at,check_cancel
from .scene_setup import TOOLS,save_scene


class SceneCanvas(Canvas):
    changed=C.Signal()

    def __init__(self,rgb,entries):
        super().__init__();self.rgb=rgb;self.entries=entries;self.tool='court';self.drag=None;self.suggestions={}
        self.setDragMode(W.QGraphicsView.DragMode.NoDrag);self.redraw()

    def redraw(self):
        self.show_frame(self.rgb)
        pix=self.item.pixmap().copy();painter=G.QPainter(pix);painter.setRenderHint(G.QPainter.RenderHint.Antialiasing)
        r=self.ratio
        painter.setPen(G.QPen(G.QColor('#a6dfef'),1,G.Qt.PenStyle.DashLine))
        for a,b in self.suggestions.get('lines',[]):painter.drawLine(C.QPointF(a[0]*r,a[1]*r),C.QPointF(b[0]*r,b[1]*r))
        for row in self.suggestions.get('people',[]):
            a,b,c,d=row['bbox'];painter.drawRect(C.QRectF(a*r,b*r,(c-a)*r,(d-b)*r))
        painter.setPen(G.QPen(G.QColor('#69ed41'),2,G.Qt.PenStyle.DashLine))
        for row in self.suggestions.get('balls',[]):
            a,b,c,d=row['bbox'];painter.drawEllipse(C.QRectF(a*r,b*r,(c-a)*r,(d-b)*r))
        for key,points in self.entries.items():
            if not points:continue
            label,color,kind=TOOLS[key];painter.setPen(G.QPen(G.QColor(color),3 if key==self.tool else 2))
            vertices=[C.QPointF(x*r,y*r) for x,y in points];painter.setBrush(C.Qt.BrushStyle.NoBrush)
            if kind in ('polygon','line','segment'):
                path=G.QPainterPath(vertices[0])
                for point in vertices[1:]:path.lineTo(point)
                if kind=='polygon' and len(vertices)>2:path.closeSubpath()
                painter.drawPath(path)
            elif len(points)==2 and kind=='rectangle':painter.drawRect(C.QRectF(vertices[0],vertices[1]).normalized())
            elif len(points)==2 and kind=='circle':
                radius=math.dist(points[0],points[1])*r;painter.drawEllipse(vertices[0],radius,radius)
            for point in vertices:
                painter.setBrush(G.QColor(color));painter.drawEllipse(point,4,4)
            caption=label if key==self.tool else {'team_a':'Наша А','team_b':'Соперник Б','ball':'Мяч','referee':'Судья'}.get(key)
            if caption:
                metrics=painter.fontMetrics();width=metrics.horizontalAdvance(caption)
                x=max(2,min(vertices[0].x()+7,pix.width()-width-2));y=max(metrics.height(),vertices[0].y()-7)
                painter.drawText(C.QPointF(x,y),caption)
        painter.end();self.item.setPixmap(pix)

    def image_point(self,event):
        p=self.mapToScene(event.position().toPoint())
        return [float(np.clip(p.x()/self.ratio,0,self.rgb.shape[1])),float(np.clip(p.y()/self.ratio,0,self.rgb.shape[0]))]

    def add_point(self,point):
        points=self.entries.setdefault(self.tool,[]);kind=TOOLS[self.tool][2]
        limit=1 if kind=='point' else 2 if kind in ('segment','circle','rectangle') else None
        if limit and len(points)>=limit:return
        points.append(point);self.redraw();self.changed.emit()

    def mousePressEvent(self,event):
        if event.button()!=C.Qt.MouseButton.LeftButton:return super().mousePressEvent(event)
        point=self.image_point(event);tolerance=10/max(.01,self.transform().m11()*self.ratio)
        nearest=None
        for key,points in self.entries.items():
            for i,p in enumerate(points):
                distance=math.dist(p,point)
                if distance<=tolerance and (nearest is None or distance<nearest[0]):nearest=(distance,key,i)
        if nearest:self.drag=(nearest[1],nearest[2]);return
        # A suggested segment becomes a typed floor line only after this click.
        if TOOLS[self.tool][2]=='line' and not self.entries.get(self.tool):
            for a,b in self.suggestions.get('lines',[]):
                v=np.asarray(b,float)-a;length=float(np.dot(v,v))
                u=np.clip(np.dot(np.asarray(point)-a,v)/max(1,length),0,1)
                if np.linalg.norm(np.asarray(point)-a-u*v)<tolerance:
                    self.entries[self.tool]=copy.deepcopy([a,b]);self.redraw();self.changed.emit();return
        self.add_point(point)

    def mouseMoveEvent(self,event):
        if self.drag:
            key,i=self.drag;self.entries[key][i]=self.image_point(event);self.redraw();self.changed.emit();return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if self.drag:self.drag=None;return
        super().mouseReleaseEvent(event)


class SceneDialog(W.QDialog):
    def __init__(self,project,source,rgb,timestamp=0,parent=None):
        super().__init__(parent);self.project=project;self.source=source;self.rgb=rgb;self.timestamp=timestamp;self.worker=None
        self.setWindowTitle('Разметка видео — всё необязательно');self.resize(1250,850)
        old=project.get('scene_setup',source['id']) or {}
        self.entries=copy.deepcopy(old.get('entries',{}))
        if old and abs(old.get('time',0)-timestamp)>.1:
            for key in ('person_a','person_b','ball'):self.entries.pop(key,None)
        layout=W.QVBoxLayout(self)
        note=W.QLabel('Выберите элемент справа и ставьте точки на кадре. Точки можно перетаскивать. '
            'Контуры замыкаются автоматически; сетка допускает любое число точек. Невидимые углы не нужны. '
            'Можно сохранить часть разметки или пропустить её целиком.')
        note.setWordWrap(True);layout.addWidget(note)
        row=W.QHBoxLayout();layout.addLayout(row);row.addWidget(W.QLabel('Кадр, секунда'))
        self.time=W.QDoubleSpinBox();self.time.setDecimals(3);self.time.setRange(0,source['duration']);self.time.setValue(timestamp);row.addWidget(self.time)
        button('Показать кадр',self.seek,row)
        split=W.QSplitter();layout.addWidget(split,1);self.canvas=SceneCanvas(rgb,self.entries);split.addWidget(self.canvas)
        right=W.QWidget();panel=W.QVBoxLayout(right);split.addWidget(right);right.setMaximumWidth(350)
        self.tools=W.QListWidget()
        for key,(label,color,kind) in TOOLS.items():
            icon=G.QPixmap(14,14);icon.fill(G.QColor(color))
            item=W.QListWidgetItem(G.QIcon(icon),label);item.setData(C.Qt.ItemDataRole.UserRole,key);self.tools.addItem(item)
        panel.addWidget(self.tools,1);self.tools.currentRowChanged.connect(self.select_tool);self.tools.setCurrentRow(0)
        self.hint=W.QLabel();self.hint.setWordWrap(True);panel.addWidget(self.hint)
        button('Убрать последнюю точку',self.undo,panel);button('Очистить выбранный элемент',self.clear,panel)
        self.ai_button=button('Найти людей, мяч и возможные линии',self.suggest,panel)
        self.ball_button=button('Использовать найденный мяч',self.use_ball,panel);self.ball_button.setEnabled(False)
        self.ai_note=W.QLabel('ИИ предлагает людей и мяч. Возможные линии подсвечиваются без назначения. '
            'Антенны и контур сетки пока задаются вручную; качество на реальном матче ещё не измерено.')
        self.ai_note.setWordWrap(True);panel.addWidget(self.ai_note)
        self.status=W.QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        row=W.QHBoxLayout();layout.addLayout(row);button('Сохранить и продолжить',self.save,row);button('Пропустить',self.reject,row)
        self.select_tool();self.canvas.changed.connect(self.update_hint)

    def select_tool(self,*args):
        item=self.tools.currentItem()
        if item:self.canvas.tool=item.data(C.Qt.ItemDataRole.UserRole);self.canvas.redraw()
        if hasattr(self,'hint'):self.update_hint()

    def update_hint(self):
        kind=TOOLS[self.canvas.tool][2]
        text={'polygon':'Обведите контур тремя или большим числом точек.',
            'line':'Отметьте видимый отрезок двумя или большим числом точек.',
            'segment':'Две точки: нижний и верхний конец антенны.',
            'circle':'Сначала центр мяча, затем точка на его краю.',
            'rectangle':'Два противоположных угла рамки судьи.',
            'point':'Один клик внутри половины площадки этой команды.',
            'points':'Можно указать нескольких игроков этой команды на данном кадре.'}[kind]
        self.hint.setText(text+' Точек: '+str(len(self.entries.get(self.canvas.tool,[]))))

    def undo(self):
        points=self.entries.get(self.canvas.tool,[])
        if points:points.pop()
        self.canvas.redraw();self.update_hint()

    def clear(self):
        self.entries.pop(self.canvas.tool,None);self.canvas.redraw();self.update_hint()

    def seek(self):
        if self.worker:return
        try:
            frame=frame_at(self.project.resolve(self.source),self.source,self.time.value())
            if abs(frame.time-self.timestamp)>.01:
                for key in ('person_a','person_b','ball'):self.entries.pop(key,None)
            self.rgb=frame.rgb;self.timestamp=frame.time;self.canvas.rgb=self.rgb;self.canvas.suggestions={};self.canvas.redraw()
            self.status.setText('Кадр выбран. Метки движущихся игроков и мяча на новом кадре нужно указать заново.');self.update_hint()
        except (ValueError,FileNotFoundError,StopIteration) as exc:self.status.setText(str(exc))

    def suggest(self):
        if self.worker:return
        rgb=self.rgb.copy()
        def task(project,progress,cancel):
            from .vision import Detector,DEFAULTS
            progress(0,'Нейросеть ищет людей и мяч на выбранном кадре…');detector=Detector(project.root/'models')
            people=detector.predict(rgb,.65,(1,));check_cancel(cancel)
            balls=detector.balls(rgb,DEFAULTS);check_cancel(cancel)
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            lines=cv2.HoughLinesP(cv2.Canny(gray,70,160),1,np.pi/180,80,
                minLineLength=max(30,rgb.shape[1]//10),maxLineGap=20)
            segments=[np.asarray(line).reshape(2,2).tolist() for line in ([] if lines is None else lines[:30])]
            return json.dumps({'people':people,'balls':balls,'lines':segments})
        self.worker=Worker(self.project.root,task);self.ai_button.setEnabled(False)
        self.worker.signals.progress.connect(lambda v,text:self.status.setText(text))
        self.worker.signals.finished.connect(self.suggested);self.worker.signals.failed.connect(self.suggest_failed)
        C.QThreadPool.globalInstance().start(self.worker)

    def suggested(self,result):
        self.worker=None;self.ai_button.setEnabled(True)
        try:self.canvas.suggestions=json.loads(result)
        except ValueError:self.status.setText(result);return
        self.canvas.redraw();count=len(self.canvas.suggestions.get('balls',[]));self.ball_button.setEnabled(count==1)
        self.status.setText('Предложения показаны пунктиром. Выберите тип линии и кликните по подходящему отрезку. '
            'Найдено людей: '+str(len(self.canvas.suggestions.get('people',[])))+'; мячей: '+str(count)+'. Ручные точки сохранены.')

    def suggest_failed(self,text):
        self.worker=None;self.ai_button.setEnabled(True);self.status.setText(text)

    def use_ball(self):
        balls=self.canvas.suggestions.get('balls',[])
        if len(balls)!=1:return
        a,b,c,d=balls[0]['bbox'];x,y=(a+c)/2,(b+d)/2
        self.entries['ball']=[[x,y],[min(self.rgb.shape[1],x+max(c-a,d-b)/2),y]];self.canvas.redraw()

    def save(self):
        if self.worker:self.status.setText('Дождитесь завершения поиска или отмените его.');return
        try:save_scene(self.project,self.source,self.rgb,self.timestamp,self.entries);self.accept()
        except (ValueError,TypeError) as exc:self.status.setText(str(exc))

    def reject(self):
        if self.worker:self.worker.stop.set();self.status.setText('Поиск останавливается. Дождитесь завершения и закройте окно.');return
        super().reject()
