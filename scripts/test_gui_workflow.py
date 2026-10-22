"""Linux offscreen GUI workflow smoke. Does not test Windows multimedia/drivers."""
import json
from pathlib import Path
from PySide6 import QtWidgets as W

from volleyvision.demo import create
from volleyvision.domain import event
from volleyvision.exporting import export_all
from volleyvision.gui import CalibrationDialog,EventDialog,MainWindow

root=Path("work/gui-workflow")
if not (root/"project.sqlite").exists():create(root)
app=W.QApplication([]);window=MainWindow(root);window.show();app.processEvents()
start=window.current.time;window.step(1);assert window.current.time>start
window.step(-1);assert abs(window.current.time-start)<.002
last=window.project.db.execute("SELECT MAX(time) FROM frames WHERE source_id=?",(window.source["id"],)).fetchone()[0]
window.seek(last);assert abs(window.current.time-last)<.002
window.seek(0)
source=window.source;rgb=window.current.rgb
cal=CalibrationDialog(window.project,source,rgb)
cal.points=[[50,100],[590,100],[590,330],[50,330]];cal.world=[[0,0],[18,0],[18,9],[0,9]]
cal.controls=[([320,215],[9,4.5])];cal.regions={"visible_image":cal.points,"people_roi":cal.points,"ball_roi":[[0,0],[640,0],[640,360],[0,360]]}
cal.check();assert cal.result is not None;cal.save()
e=event(source["id"],2.5,"attack",team_id="A",player_id="A2",outcome="continue")
dialog=EventDialog(window.project,e);dialog.save();assert dialog.result()==W.QDialog.DialogCode.Accepted
window.project.save_event(dialog.value);window.refresh_events();window.show_stats()
target=export_all(window.project);assert (target/"report.html").exists()
window.grab().save("work/gui-workflow.png");window.close()
Path("work/gui-workflow-result.json").write_text(json.dumps({"frame_step":True,"calibration_saved":True,"manual_event_saved":True,"report_exported":True,"platform":"Linux offscreen; synthetic data"},indent=2))
print("GUI workflow passed")
