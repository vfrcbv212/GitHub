"""Optional temporal action training/calibration; no pretrained weights are supplied."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description="Обучение и проверка политики временных волейбольных действий")
    subs = p.add_subparsers(dest="command", required=True)
    train = subs.add_parser("train")
    train.add_argument("train", type=Path)
    train.add_argument("validation", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--hidden-size", type=int, default=48)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--layers", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=.001)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu", choices=("cpu", "auto", "cuda"))
    calibration = subs.add_parser("calibrate")
    calibration.add_argument("artifact", type=Path)
    calibration.add_argument("dataset", type=Path)
    calibration.add_argument("output", type=Path)
    calibration.add_argument("--rules", type=Path)
    calibration.add_argument("--enable-automatic", action="store_true")
    infer = subs.add_parser("infer")
    infer.add_argument("project", type=Path)
    infer.add_argument("run_id")
    infer.add_argument("artifact", type=Path)
    args = p.parse_args(argv)
    from volleyvision.temporal import train as run_train, calibrate_artifact, infer_project
    try:
        if args.command == "train":
            result = run_train(args.train, args.validation, args.output, epochs=args.epochs,
                               resume=args.resume, hidden_size=args.hidden_size, batch_size=args.batch_size, device=args.device,
                               num_layers=args.layers, learning_rate=args.learning_rate, seed=args.seed,
                               progress=lambda fraction, record: print(json.dumps(record, ensure_ascii=False), flush=True))
        elif args.command == "calibrate":
            rules = json.loads(args.rules.read_text(encoding="utf-8")) if args.rules else None
            result = calibrate_artifact(args.artifact, args.dataset, args.output, rules=rules,
                                        enable_automatic=args.enable_automatic)
        else:
            from volleyvision.store import Project
            from volleyvision.exporting import export_all
            with Project(args.project) as project:
                result = infer_project(project, args.run_id, args.artifact)
                export_all(project)
                export_all(project, mode="preliminary")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, KeyboardInterrupt) as exc:
        print(f"Временная модель: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
