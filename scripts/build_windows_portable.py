"""Assemble a Windows ZIP from official CPython + installed Windows wheels.

No cross-platform binary substitution: runtime must contain pythonw.exe and
Windows .pyd/.dll files. Wine/native Windows self-test is a separate required step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def copy_public_docs(repo, destination):
    """Ship user documentation only; private planning/PC reports stay local."""
    if destination.exists():shutil.rmtree(destination)
    destination.mkdir(parents=True)
    names=(repo/'packaging/windows/public-docs.txt').read_text(encoding='utf-8').splitlines()
    for name in names:
        if not name or name.startswith('#'):continue
        if Path(name).name!=name:raise ValueError('Invalid public document path')
        shutil.copy2(repo/'docs'/name,destination/name)
    native=repo/'docs/validation/native-windows-ci.json'
    if native.is_file():
        (destination/'validation').mkdir()
        shutil.copy2(native,destination/'validation'/native.name)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--launcher',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('work/portable/VolleyVision'))
    parser.add_argument('--zip',type=Path)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if not (args.runtime/'pythonw.exe').is_file():raise ValueError('Windows Python runtime absent')
    if args.launcher.read_bytes()[:2]!=b'MZ':raise ValueError('Windows launcher absent')
    if (out/'runtime').exists():shutil.rmtree(out/'runtime')
    if (out/'app').exists():shutil.rmtree(out/'app')
    shutil.copytree(args.runtime,out/'runtime',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Make the C++ runtime visible before Qt is imported, including CUDA probes.
    # Keep CPython's vcruntime140/140_1; copy the matching redistributable MSVCP
    # components already shipped by the official PySide6 Windows wheel.
    for name in ('msvcp140.dll','msvcp140_1.dll','msvcp140_2.dll',
                 'msvcp140_codecvt_ids.dll','concrt140.dll','vcomp140.dll'):
        original=args.runtime/'Lib'/'site-packages'/'PySide6'/name
        if original.exists():shutil.copy2(original,out/'runtime'/name)
    shutil.copytree(repo/'volleyvision',out/'app'/'volleyvision',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(repo/'packaging/windows/portable_bootstrap.py',out/'app'/'portable_bootstrap.py')
    shutil.copy2(args.launcher,out/'VolleyVision.exe')
    copy_public_docs(repo,out/'docs')
    shutil.copytree(repo/'packaging/windows',out/'build-source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(repo/'scripts/build_windows_portable.py',out/'build-source'/'build_windows_portable.py')
    shutil.copytree(args.models,out/'models',dirs_exist_ok=True,ignore=shutil.ignore_patterns('network','*.partial'))
    supplied=json.loads((out/'models'/'manifest.json').read_text(encoding='utf-8'))
    for name in ('detector','ocr'):
        item=supplied[name];file=out/'models'/item['file']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Model SHA mismatch: '+name)
    shutil.copy2(repo/'packaging/windows/START.txt',out/'НАЧАТЬ ЗДЕСЬ.txt')
    # Adjacent DLLs remain replaceable; retain upstream licenses/source notices.
    shutil.copy2(repo/'packaging/windows/THIRD_PARTY.txt',out/'THIRD_PARTY.txt')
    (out/'Демо.cmd').write_bytes('@echo off\r\nstart "" "%~dp0VolleyVision.exe" --demo\r\n'.encode('ascii'))
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
    files={str(p.relative_to(out)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in out.rglob('*') if p.is_file() and p.name!='BUILD.json' and not any(part in ('Projects','Check') for part in p.relative_to(out).parts)}
    manifest={'app_version':'0.3.0','source_commit':commit,'runtime':'CPython 3.12.10 Windows x64',
        'mode':'CPU','entrypoint':'VolleyVision.exe','offline_weights':list(supplied),
        'qt':'6.8.3 Essentials','native_windows_verified':False,'files':files}
    (out/'BUILD.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.zip:
        archive=args.zip.resolve();archive.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
            for p in sorted(out.rglob('*')):
                if p.is_file() and not any(part in ('Projects','Check','Logs') for part in p.relative_to(out).parts):
                    zipped.write(p,Path(out.name)/p.relative_to(out))
        with zipfile.ZipFile(archive) as zipped:
            if zipped.testzip() is not None:raise ValueError('ZIP CRC failed')
        print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,
            'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'zip_crc':True},indent=2))
    else:print(str(out))


if __name__=='__main__':main()
