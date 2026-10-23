"""Build machine only. Download pinned native redistributables; no user Python runtime."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,zipfile,io,shutil

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
for key,entry in json.loads((root/'packaging/windows-deps.json').read_text()).items():
 target=args.out/key;target.mkdir(exist_ok=True);archive=args.out/(key+'.zip')
 if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest()!=entry['sha256']:
  data=urllib.request.urlopen(entry['url'],timeout=120).read()
  if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('SHA mismatch: '+key)
  archive.write_bytes(data)
 with zipfile.ZipFile(archive) as z:
  if key=='directml':
   needed=['bin/x64-win/DirectML.dll']+[n for n in z.namelist() if 'license' in n.lower() and not n.endswith('/')]
   for n in needed:z.extract(n,target)
  else:z.extractall(target)
 print(key,'verified')
