"""Build-time only: verify public sources and export the native ONNX bundle."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile,io,urllib.request

POSE_URL='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-pose.pt'
POSE_SHA='1060bda4a27012060eca246f9b2adeea22eabb045a1e58f8d229be29b7ebc2ba'
BALL_URL='https://raw.githubusercontent.com/asigatchov/fast-volleyball-tracking-inference/3aad1b98b5b744afb0100418e95892a5b0c23865/models/VballNetV4c_seq9_grayscale_20260908_213829.onnx'
BALL_SHA='8eea11cc5c881fa008ec55b0f5732ab79f2e0e1833d7f586e680f3aaa98e4b8c'
REID_URL='https://raw.githubusercontent.com/Smorodov/Multitarget-tracker/2933127e75a9bfa2a93ac1f4921fec4b65d1f213/data/reid/osnet_x0_25_msmt17.onnx'
REID_SHA='bb2267b23a64675f67bffc3177b7319e3a08de56041d153562c0206699634bce'

def download(url,sha):
 data=urllib.request.urlopen(url,timeout=90).read()
 if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Download SHA mismatch: '+url)
 return data

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'models');parser.add_argument('--reuse',type=Path);args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
 if args.reuse and (args.reuse/'yolo11s-pose.onnx').exists():shutil.copyfile(args.reuse/'yolo11s-pose.onnx',out/'pose.onnx')
 else:
  from ultralytics import YOLO
  weights=out/'yolo11s-pose.pt';weights.write_bytes(download(POSE_URL,POSE_SHA))
  exported=YOLO(str(weights)).export(format='onnx',imgsz=960,opset=17,simplify=False,dynamic=False,nms=False,device='cpu')
  shutil.move(exported,out/'pose.onnx');weights.unlink()
 for source,name,sha,url in [('VballNetV4c.onnx','ball.onnx',BALL_SHA,BALL_URL),('osnet_x0_25_msmt17.onnx','reid.onnx',REID_SHA,REID_URL)]:
  data=(args.reuse/source).read_bytes() if args.reuse and (args.reuse/source).exists() else download(url,sha)
  assert hashlib.sha256(data).hexdigest()==sha;(out/name).write_bytes(data)
 if args.reuse and (args.reuse/'jersey-rec.onnx').exists():shutil.copyfile(args.reuse/'jersey-rec.onnx',out/'jersey-rec.onnx')
 else:
  meta=json.load(urllib.request.urlopen('https://pypi.org/pypi/rapidocr_onnxruntime/1.4.4/json',timeout=30));wheel=next(x for x in meta['urls'] if x['filename'].endswith('.whl'));data=download(wheel['url'],wheel['digests']['sha256'])
  with zipfile.ZipFile(io.BytesIO(data)) as archive:(out/'jersey-rec.onnx').write_bytes(archive.read('rapidocr_onnxruntime/models/ch_PP-OCRv4_rec_infer.onnx'))
 assert hashlib.sha256((out/'jersey-rec.onnx').read_bytes()).hexdigest()=='48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b'
 files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.onnx')}
 (out/'native-models.json').write_text(json.dumps({'version':'0.2.0','pose_source_sha':POSE_SHA,'files':files},indent=2));print(json.dumps(files,indent=2))

if __name__=='__main__':main()
