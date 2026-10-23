"""Assemble/check the portable native distribution after Windows execution tests."""
from pathlib import Path
import argparse,hashlib,json,zipfile,subprocess

parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();root=Path(__file__).resolve().parents[1];folder=args.folder
assert (folder/'VolleyScope.exe').exists() and (folder/'models/native-models.json').exists()
for needed in ('Qt6Core.dll','Qt6Widgets.dll','Qt6Multimedia.dll','Qt6Sql.dll','onnxruntime.dll','DirectML.dll','msvcp140.dll','vcruntime140.dll','platforms/qwindows.dll','sqldrivers/qsqlite.dll','ffmpeg/ffmpeg.exe','ffmpeg/ffprobe.exe'):
 assert (folder/needed).exists(),needed
assert not list(folder.rglob('python*.exe'))
exe=(folder/'VolleyScope.exe').read_bytes();offset=int.from_bytes(exe[60:64],'little');assert exe[:2]==b'MZ' and exe[offset:offset+4]==b'PE\0\0' and int.from_bytes(exe[offset+4:offset+6],'little')==0x8664
files={p.relative_to(folder).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file() and p.name!='PACKAGE.json'}
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
(folder/'PACKAGE.json').write_text(json.dumps({'version':'0.2.0','source_commit':commit,'runtime':'native C++20 / Qt / ONNX Runtime','python_runtime_required':False,'windows_cpu_execution_tested':True,'target_gpu_30fps_verified':False,'files':files},ensure_ascii=False,indent=2),encoding='utf-8')
args.out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(args.out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(folder.rglob('*')):
  if p.is_file():z.write(p,'VolleyScope/'+p.relative_to(folder).as_posix())
with zipfile.ZipFile(args.out) as z:
 assert z.testzip() is None
 for name,sha in files.items():assert hashlib.sha256(z.read('VolleyScope/'+name)).hexdigest()==sha,name
sha=hashlib.sha256(args.out.read_bytes()).hexdigest();args.out.with_suffix('.sha256').write_text(f'{sha}  {args.out.name}\n');print(json.dumps({'file':args.out.name,'bytes':args.out.stat().st_size,'sha256':sha,'files':len(files),'native_windows_execution_tested':True,'target_gpu_fps_verified':False},indent=2))
