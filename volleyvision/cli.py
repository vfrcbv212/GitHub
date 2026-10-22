from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .domain import validate_event
from .media import Cancelled
from .store import Project


def parser():
    p = argparse.ArgumentParser(description="VolleyVision — локальная видеоаналитика 6×6")
    subs = p.add_subparsers(dest="command", required=True)
    for command in ("init", "gui", "import", "proxy", "fetch-models", "analyze", "setup-scan", "report", "load-profile", "import-events"):
        sub = subs.add_parser(command)
        sub.add_argument("project", type=Path)
        if command == "init": sub.add_argument("--title", default="Новый матч")
        if command=='fetch-models':sub.add_argument('--no-ocr',action='store_true',help='Подготовить только детектор; режим по внешности не требует OCR')
        if command == "import":
            sub.add_argument("videos", nargs="+", type=Path)
            sub.add_argument("--gap", type=float, default=None, help="Разрыв перед частью; без аргумента неизвестен")
        if command in ("analyze", "setup-scan"):
            sub.add_argument("--config", type=Path)
            sub.add_argument("--source")
        if command in ("load-profile", "import-events"): sub.add_argument("json_file", type=Path)
        if command == "report":
            sub.add_argument("--mode", choices=("automatic", "confirmed", "preliminary"), default="automatic")
            sub.add_argument("--filters", type=Path)
            sub.add_argument("--clips", action="store_true")
            sub.add_argument("--before", type=float, default=2)
            sub.add_argument("--after", type=float, default=3)
    d = subs.add_parser("demo"); d.add_argument("destination", type=Path)
    d = subs.add_parser("diagnose"); d.add_argument("--out", type=Path)
    d = subs.add_parser("infer-actions");d.add_argument("project",type=Path);d.add_argument("run_id");d.add_argument("artifact",type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "diagnose":
            from .diagnostics import diagnose
            output = json.dumps(diagnose(), ensure_ascii=False, indent=2)
            if args.out: args.out.write_text(output, encoding="utf-8")
            print(output); return 0
        if args.command == "demo":
            from .demo import create
            print(create(args.destination)); return 0
        if args.command == "gui":
            from .gui import run
            return run(args.project)
        with Project(args.project, create=args.command=="init", title=getattr(args,"title","Новый матч")) as project:
            if args.command == "init": print(project.root)
            elif args.command == "import":
                from .media import import_video
                for video in args.videos: print(json.dumps(import_video(project,video,args.gap),ensure_ascii=False,indent=2))
            elif args.command == "proxy":
                from .media import create_proxy
                for s in project.all("source"): print(create_proxy(project,s))
            elif args.command == "fetch-models":
                from .models import fetch
                print(json.dumps(fetch(project.root/"models",include_ocr=not args.no_ocr),ensure_ascii=False,indent=2))
            elif args.command in ("analyze", "setup-scan"):
                from .vision import analyze
                config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
                for s in project.all("source"):
                    if not args.source or args.source == s["id"]:
                        print(json.dumps(analyze(project,s["id"],config, setup_only=args.command=="setup-scan"),ensure_ascii=False,indent=2))
            elif args.command == "report":
                from .exporting import export_all
                filters = json.loads(args.filters.read_text(encoding="utf-8")) if args.filters else {}
                print(export_all(project,mode=args.mode,filters=filters,clips=args.clips,before=args.before,after=args.after))
            elif args.command == "infer-actions":
                from .temporal import infer_project
                from .exporting import export_all
                print(json.dumps(infer_project(project,args.run_id,args.artifact),ensure_ascii=False,indent=2))
                export_all(project);export_all(project,mode="preliminary")
            elif args.command == "load-profile":
                values = json.loads(args.json_file.read_text(encoding="utf-8"))
                allowed = {"team","player","lineup","side","set","rally","calibration"}
                for kind, objects in values.items():
                    if kind not in allowed: raise ValueError(f"Недопустимая сущность профиля: {kind}")
                    if kind == "calibration":
                        from .calibration import validate_profile
                        validated=[]
                        for obj in objects:
                            source=project.get("source",obj.get("source_id"))
                            if not source:
                                raise ValueError("Калибровка должна принадлежать исходнику текущего проекта")
                            validated.append(validate_profile(obj,source_id=source["id"],source_sha256=source["sha256"],
                                image_size=[source["display_width"],source["display_height"]]))
                        objects=validated
                    for obj in objects: project.put(kind,obj)
            elif args.command == "import-events":
                values = json.loads(args.json_file.read_text(encoding="utf-8"))
                events = values.get("events",[]) if isinstance(values,dict) else values
                for e in events: validate_event(e); project.check_links(e)
                for e in events: project.save_event(e)
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, Cancelled, KeyboardInterrupt) as exc:
        print(f"VolleyVision: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
