"""Create a portable source/state checkpoint, excluding private inputs and weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from datetime import datetime,timezone
from pathlib import Path


def verify_checkpoint(path):
    root=Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC failed')
        names=archive.namelist()
        if any(Path(n).is_absolute() or '..' in Path(n).parts or not n.startswith('VolleyVision/') for n in names):
            raise ValueError('Unexpected archive path')
        if any(n.lower().endswith(('.pt','.pth','.onnx')) or '.venv/' in n for n in names):
            raise ValueError('Model weights or environment included')
        manifest=json.loads(archive.read('VolleyVision/CHECKPOINT.json'))
        for name,digest in manifest['files'].items():
            if hashlib.sha256(archive.read('VolleyVision/'+name)).hexdigest()!=digest:raise ValueError('Source SHA mismatch: '+name)
        with tempfile.TemporaryDirectory(prefix='verify-checkpoint-',dir=root/'work') as folder:
            folder=Path(folder);bundle=folder/'history.bundle';bundle.write_bytes(archive.read('VolleyVision/history.bundle'))
            subprocess.run(['git','init','-q',str(folder/'git')],check=True,capture_output=True)
            subprocess.run(['git','-C',str(folder/'git'),'bundle','verify',str(bundle)],check=True,capture_output=True)
            demo=[n for n in names if n.startswith('VolleyVision/example-project/')]
            if demo:
                for name in demo:
                    target=folder/Path(name).relative_to('VolleyVision');target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(archive.read(name))
                from volleyvision.store import Project
                with Project(folder/'example-project') as project:
                    for source in project.all('source'):
                        if Path(source['path']).is_absolute():raise ValueError('Demo source is not portable')
                        video=project.resolve(source)
                        if hashlib.sha256(video.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Demo video SHA mismatch')
                    if not project.get('match','match')['synthetic']:raise ValueError('Demo synthetic flag absent')
                    if not (project.root/'exports/automatic/report.html').is_file():raise ValueError('Demo report missing')
        return {'zip_integrity':True,'source_sha_manifest':True,'private_weights_excluded':True,
            'git_bundle_included':True,'git_bundle_verified':True,'portable_demo_open':bool(demo),
            'relative_video_link':bool(demo),'commit':manifest['commit']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=Path("work/VolleyVision-checkpoint.zip"));parser.add_argument("--demo",action="store_true");parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];args.out=args.out.resolve();args.out.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    for name in ("AGENTS.md","README.md","pyproject.toml","requirements-windows.txt","requirements-linux-tested.txt","config.example.json","profile.example.json",".gitignore"):
        if (root/name).exists():files.append(root/name)
    for folder in ("volleyvision","tests","scripts","docs","packaging",".github"):
        files += [f for f in (root/folder).rglob("*") if f.is_file() and (f.suffix in (".py",".ps1",".md",".json",".xml",".txt",".c",".yml",".yaml") or f.name.startswith('Dockerfile')) and "__pycache__" not in f.parts]
    commit=subprocess.run(["git","rev-parse","HEAD"],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
    bundle=root/"work"/"history.bundle"
    subprocess.run(["git","bundle","create",str(bundle),"--all"],cwd=root,check=True,capture_output=True)
    temporary=args.out.with_suffix(".partial.zip")
    with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,Path("VolleyVision")/f.relative_to(root))
        z.write(bundle,"VolleyVision/history.bundle")
        z.writestr("VolleyVision/CHECKPOINT.json",json.dumps({"created_utc":datetime.now(timezone.utc).isoformat(),"commit":commit,
            "files":{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            "excluded":[".venv","user videos","model weights","cache","credentials"],
            "resume":"Read AGENTS.md, docs/PROJECT_STATE.md and docs/TASKS.md; restore environment from README"},indent=2))
        if args.demo:
            from volleyvision.demo import create
            demo_root=root/"work"/"portable-synthetic-demo"
            if not (demo_root/"project.sqlite").exists():create(demo_root)
            from volleyvision.store import Project
            from volleyvision.exporting import export_all
            with Project(demo_root) as p:
                export_all(p)
                backup=root/"work"/"synthetic-backup.sqlite";p.backup(backup)
            z.write(backup,"VolleyVision/example-project/project.sqlite")
            for folder in ("media","exports"):
                for f in (demo_root/folder).rglob("*"):
                    if f.is_file():z.write(f,Path("VolleyVision/example-project")/f.relative_to(demo_root))
    temporary.replace(args.out)
    result={"path":str(args.out),"bytes":args.out.stat().st_size,"sha256":hashlib.sha256(args.out.read_bytes()).hexdigest()}
    if args.verify:result['verification']=verify_checkpoint(args.out)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
