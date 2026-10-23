# Third-party models and native libraries

Application/core adapters are original C++ code licensed AGPL-3.0 (root LICENSE).
No third-party application source or private media is embedded.

- Qt 6.8.3: Qt Project / The Qt Company, dynamically linked under LGPL-3.0/GPL
  module terms. Source and license information: https://www.qt.io/licensing/open-source-lgpl-obligations
  and https://download.qt.io/archive/qt/6.8/6.8.3/single/ . End users may replace
  compatible shared libraries. License texts are included in the distribution.
- ONNX Runtime 1.24.4 DirectML: Microsoft, MIT. Pinned NuGet package SHA256
  `57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681`.
  https://github.com/microsoft/onnxruntime/tree/v1.24.4
  DirectML 1.15.4 is a separate Microsoft redistributable; its package license
  accompanies the binary. NuGet SHA256
  `4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9`.
- YOLO11s-pose: Ultralytics, COCO pretrained, AGPL-3.0. Original checkpoint
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-pose.pt
  SHA256 `1060bda4a27012060eca246f9b2adeea22eabb045a1e58f8d229be29b7ebc2ba`.
  Export uses ultralytics 8.4.184, Torch 2.13.0, ONNX opset 17, fixed 960x960,
  no NMS baked into the model. Source: https://github.com/ultralytics/ultralytics
- VballNetV4c: Alexander Sigatchov, MIT. Source
  https://github.com/asigatchov/fast-volleyball-tracking-inference at
  `3aad1b98b5b744afb0100418e95892a5b0c23865`, model sequence 9 grayscale.
  SHA256 `8eea11cc5c881fa008ec55b0f5732ab79f2e0e1833d7f586e680f3aaa98e4b8c`.
  Copyright (c) 2025 Alexander Sigatchov. Upstream MIT notice is retained in LICENSES.
- OSNet x0.25 MSMT17: public pretrained body ReID model. The ONNX artifact is
  mirrored in https://github.com/Smorodov/Multitarget-tracker at
  `2933127e75a9bfa2a93ac1f4921fec4b65d1f213`, data/reid/osnet_x0_25_msmt17.onnx.
  Mirror repository Apache-2.0; upstream OSNet implementation
  https://github.com/KaiyangZhou/deep-person-reid (MIT).
  Git blob `43a689e7130f989d5b3bc7ff61442854e409fc78`, SHA256
  `bb2267b23a64675f67bffc3177b7319e3a08de56041d153562c0206699634bce`.
  This is a body appearance embedding; volleyball identity accuracy is not established.
- PP-OCRv4 recognition: PaddleOCR / RapidAI, Apache-2.0. Model distributed in
  PyPI rapidocr_onnxruntime 1.4.4; its wheel SHA is verified against the pinned
  release metadata. Only its ONNX model is shipped; the Python package is not
  a runtime dependency. Model SHA256
  `48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b`.
  https://github.com/RapidAI/RapidOCR and https://github.com/PaddlePaddle/PaddleOCR
- FFmpeg: BtbN shared GPL build from the pinned 2026-10-23 autobuild, separate
  process binaries in ffmpeg/. Build/source instructions:
  https://github.com/BtbN/FFmpeg-Builds and https://ffmpeg.org/download.html
  Windows asset hash and exact URL are in packaging/windows-deps.json.
  FFmpeg libraries are isolated from Qt's own FFmpeg backend to avoid DLL collisions.

No raw photographs, match recordings, personal roster data or galleries are
redistributed. Pretrained model training-data rights and deployment restrictions
have not been independently audited; the sources above identify provenance.
