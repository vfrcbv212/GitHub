"""Initial kit examples and optional manual scores, using local original frames."""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore as C,QtGui as G,QtWidgets as W

from .domain import uid
from .gui import Canvas,combo,button
from .kits import save_sample,set_our_team
from .scoring import save_score,remove_score,score_report
from .roster import player_label,save_player,ROLES


class RosterDialog(W.QDialog):
    def __init__(self,project,parent=None):
        super().__init__(parent);self.project=project;self.edit_id=None
        self.setWindowTitle('Составы: номера и имена необязательны');self.resize(850,650)
        if not project.all('team'):
            for tid,label in (('A','Команда А'),('B','Команда Б')):project.put('team',{'id':tid,'name':label})
        layout=W.QVBoxLayout(self);form=W.QFormLayout();layout.addLayout(form)
        self.team=combo([(t['name'],t['id']) for t in project.all('team')]);form.addRow('Команда',self.team)
        self.team_name=W.QLineEdit();form.addRow('Новое название команды',self.team_name);button('Переименовать команду',self.rename,layout)
        self.name=W.QLineEdit();form.addRow('Имя / подпись (необязательно)',self.name)
        self.number=W.QSpinBox();self.number.setRange(-1,99);self.number.setSpecialValueText('Без номера');self.number.setValue(-1);form.addRow('Номер (необязательно)',self.number)
        self.role=combo([(label,key) for key,label in ROLES.items()]);form.addRow('Амплуа',self.role)
        note=W.QLabel('Игроки без номера получают отдельные карточки. Выберите строку для исправления существующего игрока. '
                      'После предварительного прохода привяжите несколько его кадров в галерее: они станут образцами внешности.')
        note.setWordWrap(True);layout.addWidget(note)
        self.table=W.QTableWidget(0,4);self.table.setHorizontalHeaderLabels(['Команда','Игрок','Номер','Амплуа'])
        self.table.setEditTriggers(W.QAbstractItemView.EditTrigger.NoEditTriggers);self.table.setSelectionBehavior(W.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self.load);layout.addWidget(self.table,1)
        row=W.QHBoxLayout();layout.addLayout(row);button('Сохранить карточку',self.save,row);button('Новый игрок',self.new,row);button('Закрыть',self.accept,row)
        self.status=W.QLabel();layout.addWidget(self.status);self.refresh()

    def refresh(self):
        self.table.blockSignals(True);players=self.project.all('player');teams={t['id']:t['name'] for t in self.project.all('team')}
        self.table.setRowCount(len(players))
        for i,p in enumerate(players):
            for j,value in enumerate([teams.get(p['team_id'],p['team_id']),p.get('name') or p.get('label') or player_label(p),
                                      p['number'] if p.get('number') is not None else 'нет',ROLES.get(p.get('role'),'Не указано')]):
                item=W.QTableWidgetItem(str(value));item.setData(C.Qt.ItemDataRole.UserRole,p);self.table.setItem(i,j,item)
        self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch);self.table.blockSignals(False)

    def load(self):
        row=self.table.currentRow()
        if row<0 or not self.table.selectedItems():return
        p=self.table.item(row,0).data(C.Qt.ItemDataRole.UserRole);self.edit_id=p['id']
        self.team.setCurrentIndex(self.team.findData(p['team_id']));self.name.setText(p.get('name') or '')
        self.number.setValue(p['number'] if p.get('number') is not None else -1);self.role.setCurrentIndex(self.role.findData(p.get('role','unknown')))

    def new(self):
        self.edit_id=None;self.table.clearSelection();self.name.clear();self.number.setValue(-1);self.role.setCurrentIndex(self.role.findData('unknown'))

    def save(self):
        try:
            p=save_player(self.project,self.team.currentData(),None if self.number.value()==-1 else self.number.value(),self.name.text(),self.role.currentData(),self.edit_id)
            self.new();self.refresh();self.status.setText('Сохранён '+player_label(p))
        except ValueError as exc:self.status.setText(str(exc))

    def rename(self):
        if not self.team_name.text().strip():return
        t=self.project.get('team',self.team.currentData());t['name']=self.team_name.text().strip();self.project.put('team',t)
        self.team.setItemText(self.team.currentIndex(),t['name']);self.refresh()


class KitDialog(W.QDialog):
    def __init__(self,project,source=None,rgb=None,timestamp=0,parent=None):
        super().__init__(parent);self.project=project;self.source=source;self.rgb=rgb;self.timestamp=timestamp;self.points=[]
        self.setWindowTitle('Наша команда, форма и либеро');self.resize(1150,850)
        layout=W.QVBoxLayout(self)
        note=W.QLabel('Покажите форму на нескольких кадрах: два клика по противоположным углам футболки, без пола и соседнего игрока. '
            'Либеро — отдельный образец той же команды. Цвет даёт только предположение; одинаковые цвета разрешены.')
        note.setWordWrap(True);layout.addWidget(note)
        teams=project.all('team');self.ours=combo([('Пока не указана',None)]+[(t['name'],t['id']) for t in teams],project.get('match','match').get('our_team_id'))
        row=W.QFormLayout();row.addRow('Наша команда',self.ours);layout.addLayout(row)
        self.ours.currentIndexChanged.connect(lambda:set_our_team(project,self.ours.currentData()))
        self.canvas=Canvas();layout.addWidget(self.canvas,1)
        if rgb is not None:self.canvas.show_frame(rgb)
        else:layout.addWidget(W.QLabel('Для образца импортируйте видео и выберите кадр в плеере.'))
        self.canvas.clicked.connect(self.click)
        row=W.QHBoxLayout();layout.addLayout(row)
        self.team=combo([(t['name'],t['id']) for t in teams]);row.addWidget(self.team)
        self.kind=combo([('Основная форма','regular'),('Форма либеро','libero')]);row.addWidget(self.kind)
        self.player=W.QComboBox();row.addWidget(self.player);self.team.currentIndexChanged.connect(self.players);self.players()
        button('Сохранить образец',self.save,row);button('Очистить выделение',self.clear,row)
        self.samples=W.QListWidget();self.samples.setIconSize(C.QSize(90,90));self.samples.setMaximumHeight(170);layout.addWidget(self.samples)
        row=W.QHBoxLayout();layout.addLayout(row);button('Убрать выбранный образец',self.remove,row);button('Закрыть',self.accept,row)
        self.status=W.QLabel();self.status.setWordWrap(True);layout.addWidget(self.status);self.refresh()

    def players(self):
        self.player.clear();self.player.addItem('Игрок не указан',None)
        roles={'libero':'либеро','setter':'связующий','outside':'доигровщик','opposite':'диагональный','middle':'центральный','unknown':'амплуа не указано'}
        for p in self.project.all('player'):
            if p['team_id']==self.team.currentData():self.player.addItem(f"{player_label(p)} · {roles.get(p.get('role'),'амплуа не указано')}",p['id'])

    def click(self,x,y):
        if self.rgb is None or not(0<=x<self.rgb.shape[1] and 0<=y<self.rgb.shape[0]):return
        if len(self.points)==2:self.points=[]
        self.points.append([x,y]);self.redraw()

    def redraw(self):
        shapes=[]
        if len(self.points)==2:
            (a,b),(c,d)=self.points;shapes=[{'polygon':[[a,b],[c,b],[c,d],[a,d]],'label':'Образец формы'}]
        self.canvas.show_frame(self.rgb,points=self.points,polygons=shapes)

    def clear(self):
        self.points=[]
        if self.rgb is not None:self.redraw()

    def save(self):
        try:
            if self.rgb is None or not self.source:raise ValueError('Сначала выберите кадр исходного видео')
            if len(self.points)!=2:raise ValueError('Укажите два противоположных угла участка футболки')
            (a,b),(c,d)=self.points
            save_sample(self.project,self.source['id'],self.timestamp,self.rgb,[min(a,c),min(b,d),max(a,c),max(b,d)],
                self.team.currentData(),self.kind.currentData(),self.player.currentData())
            self.clear();self.refresh();self.status.setText('Образец сохранён. Можно добавить другой ракурс или освещение; личность автоматически не подтверждается.')
        except (ValueError,RuntimeError) as exc:self.status.setText(str(exc))

    def refresh(self):
        self.samples.clear();teams={t['id']:t['name'] for t in self.project.all('team')};players={p['id']:p for p in self.project.all('player')}
        for s in self.project.all('kit_sample'):
            if not s.get('active',True):continue
            p=players.get(s.get('player_id'));number=f" · {player_label(p)}" if p else ''
            item=W.QListWidgetItem(G.QIcon(str(self.project.root/s['path'])),
                f"{teams.get(s['team_id'],s['team_id'])}{number} · {'либеро' if s['kit_kind']=='libero' else 'основная'} · {s['time']:.3f} с")
            item.setData(C.Qt.ItemDataRole.UserRole,s['id']);self.samples.addItem(item)

    def remove(self):
        item=self.samples.currentItem()
        if item:
            s=self.project.get('kit_sample',item.data(C.Qt.ItemDataRole.UserRole));s['active']=False;self.project.put('kit_sample',s);self.refresh()


class ScoreDialog(W.QDialog):
    def __init__(self,project,source_id=None,timestamp=0,parent=None):
        super().__init__(parent);self.project=project;self.edit_id=None;self.teams=project.all('team')
        self.setWindowTitle('Ручной счёт — без распознавания табло');self.resize(1000,720)
        layout=W.QVBoxLayout(self);note=W.QLabel('Укажите счёт после завершённого розыгрыша или итог партии. '
            'Пропущенные отметки разрешены. Счёт не определяет автора и причину очка; анализ работает и без него.')
        note.setWordWrap(True);layout.addWidget(note)
        row=W.QHBoxLayout();layout.addLayout(row);self.party=W.QComboBox();row.addWidget(self.party)
        self.number=W.QSpinBox();self.number.setRange(1,99);row.addWidget(self.number);button('Создать / выбрать партию №',self.add_set,row)
        form=W.QFormLayout();layout.addLayout(form);self.boxes={}
        for t in self.teams:
            box=W.QSpinBox();box.setRange(0,999);self.boxes[t['id']]=box;form.addRow(t['name'],box)
        self.source_names={s['id']:s.get('name') or Path(s.get('path',s['id'])).name for s in project.all('source')}
        self.source=combo([(name,sid) for sid,name in self.source_names.items()],source_id)
        form.addRow('Часть записи',self.source);self.time=W.QDoubleSpinBox();self.time.setDecimals(3);form.addRow('Время после очка, с',self.time)
        self.source.currentIndexChanged.connect(self.time_range);self.time_range();self.time.setValue(timestamp)
        self.link_time=W.QCheckBox('Привязать к времени записи');self.link_time.setChecked(bool(source_id));form.addRow(self.link_time)
        self.final=W.QCheckBox('Итог партии');self.initial=W.QCheckBox('Начало партии: 0:0');form.addRow(self.final);form.addRow(self.initial)
        self.table=W.QTableWidget(0,5);self.table.setHorizontalHeaderLabels(['Партия','Счёт','Время','Тип','Часть записи'])
        self.table.setSelectionBehavior(W.QAbstractItemView.SelectionBehavior.SelectRows);self.table.setEditTriggers(W.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.load_selected);layout.addWidget(self.table,1)
        row=W.QHBoxLayout();layout.addLayout(row);button('Сохранить / исправить отметку',self.save,row);button('Новая отметка',self.new,row)
        button('Убрать выбранную отметку',self.remove,row);button('Закрыть',self.accept,row)
        self.status=W.QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        self.refresh_sets();self.refresh()

    def time_range(self):
        source=self.project.get('source',self.source.currentData());self.time.setRange(0,source['duration'] if source else 0)

    def refresh_sets(self,selected=None):
        chosen=selected or self.party.currentData();self.party.clear()
        parties=sorted(self.project.all('set'),key=lambda s:s.get('number',0))
        for s in parties:self.party.addItem(f"Партия {s.get('number','?')}",s['id'])
        if self.party.findData(chosen)>=0:self.party.setCurrentIndex(self.party.findData(chosen))
        self.number.setValue(max([s.get('number',0) for s in parties]+[0])+1)

    def add_set(self):
        existing=next((s for s in self.project.all('set') if s.get('number')==self.number.value()),None)
        s=existing or self.project.put('set',{'id':uid(),'number':self.number.value()});self.refresh_sets(s['id'])

    def save(self):
        try:
            save_score(self.project,self.party.currentData(),{tid:box.value() for tid,box in self.boxes.items()},
                self.source.currentData() if self.link_time.isChecked() else None,self.time.value() if self.link_time.isChecked() else None,
                self.final.isChecked(),self.initial.isChecked(),self.edit_id)
            self.new();self.refresh()
        except ValueError as exc:self.status.setText(str(exc))

    def new(self):self.edit_id=None;self.table.clearSelection();self.final.setChecked(False);self.initial.setChecked(False)

    def refresh(self):
        self.table.blockSignals(True);rows=[s for s in self.project.all('score_snapshot') if not s.get('removed')]
        self.table.setRowCount(len(rows));parties={s['id']:s for s in self.project.all('set')}
        for i,s in enumerate(rows):
            values=[parties.get(s['set_id'],{}).get('number','?'),':'.join(str(s['scores'].get(t['id'],'?')) for t in self.teams),
                f"{s['time']:.3f} с" if s.get('time') is not None else 'не задано','Итог' if s['final'] else 'Начало' if s['initial'] else 'Отметка',self.source_names.get(s.get('source_id'),'без времени')]
            for j,value in enumerate(values):
                item=W.QTableWidgetItem(str(value));item.setData(C.Qt.ItemDataRole.UserRole,s);self.table.setItem(i,j,item)
        self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch);self.table.blockSignals(False)
        report=score_report(self.project);reasons=[x['reason'] for x in report['conflicts']]
        self.status.setText('Сверка: '+('; '.join(reasons) if reasons else 'противоречий с известными розыгрышами не найдено; полнота неизвестна.'))

    def load_selected(self):
        row=self.table.currentRow()
        if row<0 or not self.table.selectedItems():return
        s=self.table.item(row,0).data(C.Qt.ItemDataRole.UserRole);self.edit_id=s['id']
        self.party.setCurrentIndex(self.party.findData(s['set_id']))
        for tid,box in self.boxes.items():box.setValue(s['scores'].get(tid,0))
        self.link_time.setChecked(s.get('time') is not None)
        if s.get('source_id'):self.source.setCurrentIndex(self.source.findData(s['source_id']));self.time.setValue(s['time'])
        self.final.setChecked(s['final']);self.initial.setChecked(s['initial'])

    def remove(self):
        if self.edit_id:remove_score(self.project,self.edit_id);self.new();self.refresh()
