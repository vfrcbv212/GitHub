"""Default compact workflow, retaining the existing advanced editors."""
from pathlib import Path
import os
import webbrowser

from PySide6 import QtCore as C,QtWidgets as W

from .gui import MainWindow,button
from .media import frame_at
from .roster import identity_mode,set_identity_mode


class SimpleWindow(MainWindow):
    def __init__(self,root):
        super().__init__(root)
        self.setWindowTitle('VolleyVision — видео → разметка → отчёт')
        self.resize(1320,850);self._import_before=None
        outer=self.centralWidget().layout();banner=outer.itemAt(0).widget()
        banner.setText('Добавьте видео. При желании отметьте площадку, сетку, мяч и команды. Затем запустите анализ. '
            'Автоматические личности и действия ещё требуют проверки на реальном видео.')
        top=outer.itemAt(1).layout();self.advanced_top=[top.itemAt(i).widget() for i in (3,4)]
        self.advanced_button=W.QPushButton('Дополнительно');self.advanced_button.setCheckable(True);top.addWidget(self.advanced_button)
        self.quick=W.QWidget();layout=W.QVBoxLayout(self.quick);self.tabs.insertTab(0,self.quick,'Начало')
        title=W.QLabel('1. Видео\n\nИспользуйте «Импорт видео» сверху. Для каждого нового файла откроется необязательная разметка.\n\n'
            '2. Разметка по желанию');title.setWordWrap(True);layout.addWidget(title)
        self.scene_button=button('Разметить кадр: поле, сетка, мяч и команды',self.scene_setup,layout)
        button('Команды и игроки — необязательно',self.roster,layout)
        layout.addWidget(W.QLabel('Как узнавать игроков'))
        layout.addWidget(self.identification)
        note=W.QLabel('Можно отметить только нужные элементы. Игроков без имён и номеров анализ не блокирует. '
            'Новые траектории не получают прежнюю личность автоматически.');note.setWordWrap(True);layout.addWidget(note)
        layout.addStretch()
        layout.addWidget(W.QLabel('3. Обработка'))
        if os.environ.get('VOLLEYVISION_BUNDLE_ROOT'):
            button('Подготовить ускорение NVIDIA',self.prepare_nvidia,layout)
        self.analyze_button=button('Анализировать видео → получить отчёт',lambda:self.analysis(False),layout)
        button('Открыть готовый HTML-отчёт',self.open_report,layout)
        self.advanced_button.toggled.connect(self.set_advanced);self.set_advanced(False);self.tabs.setCurrentWidget(self.quick)

    def set_advanced(self,enabled):
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) in ('Настройка','Эпизоды (необязательно)'):self.tabs.setTabVisible(i,enabled)
        for widget in self.advanced_top:widget.setVisible(enabled)
        self.filters.setVisible(enabled)
        self.advanced_button.setText('Скрыть дополнительные функции' if enabled else 'Дополнительно')
        if not enabled and self.tabs.currentWidget() is not self.quick and self.tabs.tabText(self.tabs.currentIndex())!='Отчёт':
            self.tabs.setCurrentWidget(self.quick)

    def open_project(self):
        super().open_project()
        if not self.project.get('identity_config','default'):
            set_identity_mode(self.project,'appearance');self.refresh()

    def import_files(self):
        if self.worker:return
        self._import_before={s['id'] for s in self.project.all('source')}
        super().import_files()
        if not self.worker:self._import_before=None

    def done(self,result):
        imported=self._import_before;self._import_before=None
        super().done(result)
        if imported is not None:
            added=[s for s in self.project.all('source') if s['id'] not in imported]
            if added:
                self.sources.setCurrentIndex(self.sources.findData(added[0]['id']))
                C.QTimer.singleShot(0,self.scene_setup)
        elif str(result).startswith('Анализ завершён'):
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i)=='Отчёт':self.tabs.setCurrentIndex(i);break

    def scene_setup(self):
        if self.worker:self.message.setText('Дождитесь текущей операции или отмените её');return
        if not self.source or not self.current:self.message.setText('Сначала импортируйте видео');return
        try:
            from .quick_setup_ui import SceneDialog
            frame=frame_at(self.project.resolve(self.source),self.source,self.current.time)
            SceneDialog(self.project,self.source,frame.rgb,frame.time,self).exec()
        except (ValueError,RuntimeError,FileNotFoundError) as exc:self.error(exc)

    def open_report(self):
        path=self.project.root/'exports'/'automatic'/'report.html'
        if path.is_file():webbrowser.open(path.resolve().as_uri())
        else:self.message.setText('Отчёт появится после завершения анализа')
