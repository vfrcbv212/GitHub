"""Initial roster confirmation, separate from automatic identity inference.

The gallery can leave every fragment unknown. Manual confirmation applies only
to an explicitly selected interval; an OCR number or a suggested match never
silently binds another track to that person.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .domain import uid
from .store import dumps
from .roster import player_label as roster_label,identity_mode

ROLES = {
    "participant": "Игрок на площадке",
    "referee": "Судья",
    "outside": "Запасной / зритель",
    "unknown": "Роль неизвестна",
}


class BindingConflict(ValueError):
    """A manual assignment would put one player on two simultaneous tracks."""


def _overlap(a, b):
    start, end = max(a["start"], b["start"]), min(a["end"], b["end"])
    if start < end - 1e-9:
        return True
    # A single observed frame is a valid interval, but adjacent positive-duration
    # fragments may meet at their boundary without implying two simultaneous people.
    return abs(start - end) <= 1e-9 and (
        a["start"] == a["end"] or b["start"] == b["end"]
    )


def effective_bindings(project):
    """Resolve last-write interval corrections without discarding their history."""
    grouped = defaultdict(list)
    for binding in project.all("track_binding"):
        grouped[(binding["run_id"], binding["track_id"])].append(binding)
    result = []
    for bindings in grouped.values():
        cuts = sorted({v for b in bindings for v in (b["start"], b["end"])})
        intervals = list(zip(cuts, cuts[1:])) or [(cuts[0], cuts[0])]
        for start, end in intervals:
            midpoint = (start + end) / 2
            covering = [b for b in bindings if b["start"] <= midpoint <= b["end"]]
            if covering:
                result.append(covering[-1] | {"start": start, "end": end})
        for b in bindings:
            if b["start"] == b["end"] and project.track_identity(b["run_id"], b["track_id"], b["start"])["id"] == b["id"]:
                if not any(x["id"] == b["id"] for x in result):
                    result.append(b)
    return result


def assignment_conflicts(project, proposed, ignored_tracks=()):
    """Return effective same-player overlaps in the same source and analysis run."""
    if proposed.get("role") != "participant" or not proposed.get("player_id"):
        return []
    tracks = {t["id"]: t for t in project.all("track")}
    ignored = set(ignored_tracks) | {proposed["track_id"]}
    conflicts = []
    for binding in effective_bindings(project):
        track = tracks.get(binding["track_id"])
        if (
            binding["track_id"] not in ignored
            and track
            and binding["run_id"] == proposed["run_id"]
            and track["source_id"] == proposed["source_id"]
            and binding.get("role") == "participant"
            and binding.get("player_id") == proposed["player_id"]
            and _overlap(binding, proposed)
        ):
            conflicts.append(binding)
    return conflicts


def _assignment(project, track_id, start, end, player_id, team_id, role):
    track = project.get("track", track_id)
    if not track:
        raise ValueError("Выбранная траектория больше не существует")
    if role not in ROLES or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end)):
        raise ValueError("Проверьте роль и конечные значения времени")
    if start > end or start < track["start"] - 1e-6 or end > track["end"] + 1e-6:
        raise ValueError("Интервал привязки должен находиться внутри выбранной траектории")
    if role != "participant":
        player_id, team_id = None, None
    if team_id is not None and not project.get("team", team_id):
        raise ValueError("Команда отсутствует в составе")
    if player_id is not None:
        player = project.get("player", player_id)
        if not player or (team_id is not None and player["team_id"] != team_id):
            raise ValueError("Выбранный игрок не принадлежит указанной команде")
        team_id = player["team_id"]
    return {
        "id": uid(), "run_id": track["run_id"], "track_id": track_id,
        "source_id": track["source_id"], "start": float(start), "end": float(end),
        "player_id": player_id, "team_id": team_id, "role": role,
        "origin": "manual_initial_gallery", "review": "confirmed",
        "identity_status": "confirmed" if player_id is not None else "unknown",
        "identity_reason": "явно выбранный игрок и интервал" if player_id is not None else "личность не установлена",
        "sample_evidence": [{"path": s.get("path"), "time": s.get("time")} for s in (track.get('person_samples') or track.get("samples", []))
                            if start<=s.get('time',track['start'])<=end],
    }


def assign_interval(project, track_id, start, end, player_id=None, team_id=None, role="participant"):
    proposed = _assignment(project, track_id, start, end, player_id, team_id, role)
    conflicts = assignment_conflicts(project, proposed)
    if conflicts:
        ids = ", ".join(dict.fromkeys(b["track_id"] for b in conflicts))
        raise BindingConflict(f"В это время игрок уже привязан к другой траектории: {ids}. Измените интервал или оставьте личность неизвестной.")
    return project.put("track_binding", proposed)


def assign_fragments(project, track_ids, player_id=None, team_id=None, role="participant"):
    """Atomically bind explicitly selected, non-overlapping fragments to a roster ID."""
    track_ids = list(dict.fromkeys(track_ids))
    proposed = []
    for track_id in track_ids:
        track = project.get("track", track_id)
        if not track:
            raise ValueError("Один из выбранных фрагментов больше не существует")
        proposed.append(_assignment(project, track_id, track["start"], track["end"], player_id, team_id, role))
    for i, candidate in enumerate(proposed):
        conflicts = assignment_conflicts(project, candidate, ignored_tracks=track_ids)
        if conflicts:
            raise BindingConflict("Игрок уже занят другой траекторией на части выбранного времени; привязки не изменены")
        if candidate.get("player_id") and candidate["role"] == "participant":
            for other in proposed[:i]:
                if candidate["run_id"] == other["run_id"] and candidate["source_id"] == other["source_id"] and _overlap(candidate, other):
                    raise BindingConflict("Выбранные фрагменты пересекаются по времени. Объединение не выполнено; уточните отдельные интервалы.")
    with project.db:
        project.db.executemany("INSERT INTO objects(kind,id,body) VALUES('track_binding',?,?)",
                              [(b["id"], dumps(b)) for b in proposed])
    return proposed


def existing_conflicts(project, bindings=None):
    """Expose old contradictory assignments without stopping unattended analysis."""
    bindings = effective_bindings(project) if bindings is None else bindings
    tracks = {t["id"]: t for t in project.all("track")}
    result = []
    for i, a in enumerate(bindings):
        if not a.get("player_id") or a.get("role") != "participant":
            continue
        for b in bindings[:i]:
            if (b.get("role") == "participant" and a["player_id"] == b.get("player_id")
                    and a["track_id"] != b["track_id"] and a["run_id"] == b["run_id"]
                    and a["track_id"] in tracks and b["track_id"] in tracks
                    and tracks[a["track_id"]]["source_id"] == tracks[b["track_id"]]["source_id"]
                    and _overlap(a, b)):
                result.append({"player_id": a["player_id"], "track_ids": [a["track_id"], b["track_id"]],
                               "start": max(a["start"], b["start"]), "end": min(a["end"], b["end"])})
    return result


def _track_incomplete(track, bindings):
    known = [b for b in bindings if b["run_id"] == track["run_id"] and b["track_id"] == track["id"]
             and (b.get("player_id") or b.get("role") in ("referee", "outside"))]
    duration = track["end"] - track["start"]
    covered = sum(max(0, min(b["end"], track["end"]) - max(b["start"], track["start"])) for b in known)
    return (duration <= 1e-9 and not known) or covered < duration - 1e-6


def gallery_summary(project):
    players = project.all("player")
    tracks = project.all("track")
    bindings = effective_bindings(project)
    assigned = {b["player_id"] for b in bindings if b.get("player_id") and b.get("role") == "participant"}
    unknown_tracks = 0
    for track in tracks:
        if _track_incomplete(track, bindings):
            unknown_tracks += 1
    return {"players_confirmed": len(assigned), "players_total": len(players), "tracks_total": len(tracks),
            "unknown_tracks": unknown_tracks, "conflicts": existing_conflicts(project, bindings),
            "players_without_samples": [p["id"] for p in players if p["id"] not in assigned]}


class GalleryDialog(QtWidgets.QDialog):
    """Multi-image confirmation dialog; optional before an unattended analysis."""

    previewRequested = QtCore.Signal(str, float)
    bindingsChanged = QtCore.Signal()

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Начальная галерея игроков")
        self.resize(1160, 830)
        self.tracks = {t["id"]: t for t in project.all("track")}
        self.players = {p["id"]: p for p in project.all("player")}
        self.teams = {t["id"]: t for t in project.all("team")}
        self.current_track = None
        layout = QtWidgets.QVBoxLayout(self)
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        intro = QtWidgets.QLabel("Подтвердите личность по нескольким кадрам. OCR и предположения не назначают игрока автоматически. Неизвестные игроки не блокируют запуск анализа.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        filters = QtWidgets.QHBoxLayout()
        layout.addLayout(filters)
        self.source_filter = QtWidgets.QComboBox()
        self.source_filter.addItem("Все части записи", None)
        for source in project.all("source"):
            self.source_filter.addItem(Path(source["path"]).name, source["id"])
        filters.addWidget(self.source_filter)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Поиск по фрагменту, номеру или игроку")
        filters.addWidget(self.search, 1)
        self.only_unknown = QtWidgets.QCheckBox("Без полной привязки")
        filters.addWidget(self.only_unknown)
        splitter = QtWidgets.QSplitter()
        layout.addWidget(splitter, 1)
        self.track_list = QtWidgets.QListWidget()
        self.track_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.track_list.setMinimumWidth(340)
        splitter.addWidget(self.track_list)
        details = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(details)
        self.sample_list = QtWidgets.QListWidget()
        self.sample_list.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.sample_list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.sample_list.setMovement(QtWidgets.QListView.Movement.Static)
        self.sample_list.setIconSize(QtCore.QSize(160, 160))
        self.sample_list.setGridSize(QtCore.QSize(190, 205))
        self.sample_list.setMinimumHeight(220)
        detail_layout.addWidget(self.sample_list)
        self.evidence = QtWidgets.QTextBrowser()
        self.evidence.setMaximumHeight(180)
        detail_layout.addWidget(self.evidence)
        form = QtWidgets.QFormLayout()
        detail_layout.addLayout(form)
        self.team = QtWidgets.QComboBox()
        self.team.addItem("Команда неизвестна", None)
        for team in self.teams.values():
            self.team.addItem(team["name"], team["id"])
        form.addRow("Команда", self.team)
        self.player = QtWidgets.QComboBox()
        form.addRow("Игрок", self.player)
        self.role = QtWidgets.QComboBox()
        for key, label in ROLES.items():
            self.role.addItem(label, key)
        form.addRow("Роль человека", self.role)
        self.start = QtWidgets.QDoubleSpinBox()
        self.end = QtWidgets.QDoubleSpinBox()
        for widget in (self.start, self.end):
            widget.setDecimals(6)
            widget.setRange(0, 1e9)
        form.addRow("Начало в исходнике, с", self.start)
        form.addRow("Конец в исходнике, с", self.end)
        buttons = QtWidgets.QHBoxLayout()
        detail_layout.addLayout(buttons)
        self.assign_button = QtWidgets.QPushButton("Подтвердить этот интервал")
        buttons.addWidget(self.assign_button)
        self.fragments_button = QtWidgets.QPushButton("Связать выбранные фрагменты целиком")
        buttons.addWidget(self.fragments_button)
        self.unknown_button = QtWidgets.QPushButton("Оставить личность неизвестной")
        detail_layout.addWidget(self.unknown_button)
        self.preview_button = QtWidgets.QPushButton("Перейти к выбранному кадру в видео")
        detail_layout.addWidget(self.preview_button)
        splitter.addWidget(details)
        splitter.setStretchFactor(1, 1)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        close = QtWidgets.QPushButton("Готово — можно продолжить анализ с неизвестными игроками")
        close.clicked.connect(self.accept)
        layout.addWidget(close)
        self.source_filter.currentIndexChanged.connect(self.reload_tracks)
        self.search.textChanged.connect(self.reload_tracks)
        self.only_unknown.toggled.connect(self.reload_tracks)
        self.track_list.currentItemChanged.connect(self.select_track)
        self.team.currentIndexChanged.connect(self.populate_players)
        self.role.currentIndexChanged.connect(self.role_changed)
        self.assign_button.clicked.connect(self.save_interval)
        self.fragments_button.clicked.connect(self.save_fragments)
        self.unknown_button.clicked.connect(self.leave_unknown)
        self.preview_button.clicked.connect(self.preview_sample)
        self.sample_list.itemDoubleClicked.connect(self.preview_sample)
        self.populate_players()
        self.reload_tracks()

    def player_label(self, player_id):
        player = self.players.get(player_id)
        if not player:
            return "Личность неизвестна"
        team = self.teams.get(player["team_id"], {}).get("name", player["team_id"])
        return f'{team}, {roster_label(player)}'

    def populate_players(self):
        selected = self.player.currentData()
        self.player.clear()
        self.player.addItem("Игрок неизвестен", None)
        team_id = self.team.currentData()
        if team_id:
            for player in self.players.values():
                if player["team_id"] == team_id:
                    self.player.addItem(self.player_label(player["id"]), player["id"])
        index = self.player.findData(selected)
        if index >= 0:
            self.player.setCurrentIndex(index)

    def role_changed(self):
        participant = self.role.currentData() == "participant"
        self.team.setEnabled(participant)
        self.player.setEnabled(participant)

    def reload_tracks(self):
        selected = self.current_track["id"] if self.current_track else None
        summary = gallery_summary(self.project)
        self.summary_label.setText(
            f"Подтверждены {summary['players_confirmed']} из {summary['players_total']} игроков состава. "
            f"Фрагментов: {summary['tracks_total']}; личность или часть интервала неизвестна: {summary['unknown_tracks']}. "
            f"Противоречивых пересечений: {len(summary['conflicts'])}. "
            "Персональная статистика непривязанных игроков может быть неполной."
        )
        self.track_list.clear()
        query = self.search.text().casefold().strip()
        source_id = self.source_filter.currentData()
        active = effective_bindings(self.project)
        selected_item = None
        for track in sorted(self.tracks.values(), key=lambda t: (t["source_id"], t["start"], t["id"])):
            if source_id and track["source_id"] != source_id:
                continue
            bindings = [b for b in active if b["run_id"] == track["run_id"] and b["track_id"] == track["id"]]
            labels = list(dict.fromkeys(self.player_label(b.get("player_id")) for b in bindings))
            full_known = not _track_incomplete(track, bindings)
            if self.only_unknown.isChecked() and full_known:
                continue
            number = track.get("number_consensus", {}).get("number")
            text = f"{track['id']}\n{track['start']:.3f}–{track['end']:.3f} с"
            text += ' · по внешности' if identity_mode(self.project)=='appearance' else f" · OCR №{number if number is not None else '?'}"
            text += "\n" + ("; ".join(labels) if labels else "Личность неизвестна")
            if any(track["id"] in c["track_ids"] for c in summary["conflicts"]):
                text += "\n⚠ Одновременная привязка одного игрока к разным траекториям"
            if query and query not in text.casefold():
                continue
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, track["id"])
            for sample in track.get('person_samples') or track.get("samples", []):
                path = Path(sample["path"])
                path = path if path.is_absolute() else self.project.root / path
                if path.is_file():
                    item.setIcon(QtGui.QIcon(str(path)))
                    break
            self.track_list.addItem(item)
            if track["id"] == selected:
                selected_item = item
        if selected_item:
            self.track_list.setCurrentItem(selected_item)
        elif self.track_list.count():
            self.track_list.setCurrentRow(0)
        else:
            self.current_track = None
            self.sample_list.clear()
            self.evidence.setPlainText("Нет подходящих фрагментов. Непривязанные игроки не блокируют запуск.")

    def select_track(self, current, previous=None):
        if not current:
            return
        track = self.tracks[current.data(QtCore.Qt.ItemDataRole.UserRole)]
        self.current_track = track
        self.start.setRange(track["start"], track["end"])
        self.end.setRange(track["start"], track["end"])
        self.start.setValue(track["start"])
        self.end.setValue(track["end"])
        self.sample_list.clear()
        for sample in track.get('person_samples') or track.get("samples", []):
            t = sample.get("time", track["start"])
            item = QtWidgets.QListWidgetItem(f"{t:.6f} с")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, t)
            path = Path(sample.get("path", ""))
            path = path if path.is_absolute() else self.project.root / path
            if path.is_file():
                item.setIcon(QtGui.QIcon(str(path)))
            else:
                item.setText(item.text() + "\nИзображение недоступно")
            item.setToolTip("Двойной щелчок — точный переход к исходному видео")
            self.sample_list.addItem(item)
        if self.sample_list.count():
            self.sample_list.setCurrentRow(0)
        consensus = track.get("number_consensus", {})
        lines = [f"Предполагаемый номер: {consensus.get('number') if consensus.get('number') is not None else 'не определён'}",
                 f"Голоса OCR: {consensus.get('votes', {})}; {consensus.get('reason', 'наблюдений нет')}",
                 "Оценки OCR не являются калиброванными вероятностями; номер не определяет команду."]
        if identity_mode(self.project)=='appearance':lines=['Режим по внешности: номера не используются. Образцы человека сохраняются между выходами из кадра.']
        proposal=track.get('team_proposal',{})
        if proposal.get('team_id'):
            lines.append('Команда траектории: '+self.teams.get(proposal['team_id'],{}).get('name',proposal['team_id'])+
                         (' — указана пользователем' if proposal.get('evidence')=='confirmed' else ' — предположение')+
                         '. Личность этим не определяется.')
        for observation in track.get("ocr_observations", [])[-20:]:
            if identity_mode(self.project)=='appearance':break
            lines.append(f"{observation['time']:.3f} с: №{observation.get('number', '?')}, score={observation.get('raw_score', 'нет')}; {observation.get('reason', '')}")
        hypotheses = [h for h in self.project.all("identity_hypothesis") if h["track_id"] == track["id"] and
                      h.get('mode','hybrid')==identity_mode(self.project)]
        for h in self.project.all('kit_hypothesis'):
            if h['track_id']==track['id']:
                lines.append('Форма: '+h['reason'])
                for c in h['candidates'][:2]:
                    lines.append(f"Команда {self.teams.get(c['team_id'],{}).get('name',c['team_id'])}: расстояние образцов {c['appearance_distance']:.3f}; предположение, не подтверждение")
        if hypotheses:
            lines.append("Предположения второго прохода — не подтверждённые привязки:")
            for hypothesis in hypotheses:
                lines.append(hypothesis.get('rule',''))
                for candidate in hypothesis.get("candidates", []):
                    lines.append(f"• {self.player_label(candidate.get('player_id'))}; расстояние образцов {candidate.get('appearance_distance', '?')}")
        self.evidence.setPlainText("\n".join(lines))
        binding = self.project.track_identity(track["run_id"], track["id"], (track["start"] + track["end"]) / 2)
        self.team.setCurrentIndex(0)
        self.role.setCurrentIndex(self.role.findData(binding.get("role", "participant") if binding else "participant"))
        if binding:
            player = self.players.get(binding.get("player_id"))
            team_id = player["team_id"] if player else binding.get("team_id")
            team_index = self.team.findData(team_id)
            if team_index >= 0:
                self.team.setCurrentIndex(team_index)
            player_index = self.player.findData(binding.get("player_id"))
            if player_index >= 0:
                self.player.setCurrentIndex(player_index)
        self.role_changed()

    def _saved(self, count):
        self.message.setText(f"Сохранено привязок: {count}. Изменены только явно выбранные интервалы; другие фрагменты не получили личность автоматически.")
        self.bindingsChanged.emit()
        self.reload_tracks()

    def save_interval(self):
        if not self.current_track:
            return
        try:
            assign_interval(self.project, self.current_track["id"], self.start.value(), self.end.value(),
                            self.player.currentData(), self.team.currentData(), self.role.currentData())
            self._saved(1)
        except ValueError as exc:
            self.message.setText(str(exc))

    def save_fragments(self):
        ids = [item.data(QtCore.Qt.ItemDataRole.UserRole) for item in self.track_list.selectedItems()]
        if not ids:
            return
        try:
            saved = assign_fragments(self.project, ids, self.player.currentData(), self.team.currentData(), self.role.currentData())
            self._saved(len(saved))
        except ValueError as exc:
            self.message.setText(str(exc))

    def leave_unknown(self):
        self.player.setCurrentIndex(0)
        self.team.setCurrentIndex(0)
        self.save_interval()

    def preview_sample(self, *args):
        if not self.current_track:
            return
        sample = self.sample_list.currentItem()
        t = sample.data(QtCore.Qt.ItemDataRole.UserRole) if sample else self.start.value()
        self.previewRequested.emit(self.current_track["source_id"], float(t))
