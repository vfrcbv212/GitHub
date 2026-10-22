# Модели, источники и условия

Сверка первичных источников выполнена 22.10.2026. Версии зафиксированы в
pyproject.toml, requirements-windows.txt и requirements-linux-tested.txt.
Windows-колёса Python 3.12 проверены по метаданным PyPI; запуск Windows/CUDA
не выполнялся. Точная GPU пользователя неизвестна.

| Вариант | Код | Веса/данные | Решение |
|---|---|---|---|
| TorchVision Faster R-CNN MobileNet 320 | BSD-3-Clause | COCO_V1; отдельные ограничения обучающих изображений | Подключён: стандартные колёса PyTorch, CPU/CUDA; не модель волейбольных действий |
| YOLOX | Apache-2.0 | upstream COCO weights; условия изображений отдельно | Кандидат дальнейшего сравнения; не установлен, качество на вашем видео неизвестно |
| Ultralytics | AGPL-3.0 / коммерческая лицензия | условия весов/данных отдельно | В базовом сценарии не используется; бесплатная загрузка не означает произвольное применение |
| Norfair 2.3.0 | BSD-3-Clause | весов нет; distance + фильтр состояния | Подключён, временные ID без распознавания лиц |
| EasyOCR 1.7.2 english_g2 | Apache-2.0 | отдельное явное разрешение на веса и весь обучающий набор независимо не подтверждено | Подключён для локального пилота; не объявлять права на любое распространение весов |

Предобученные веса TorchVision могут иметь условия, связанные с обучающими
данными: кодовая лицензия не заменяет проверку прав данных. COCO-аннотации
публикуются по CC BY 4.0, права изображений индивидуальны. Веса не включаются в
исходный архив; `fetch-models` загружает их непосредственно из upstream и
сохраняет manifest с источниками, версиями, ограничениями и SHA256.
Программа после этой операции не загружает ничего автоматически.

Проверенные файлы:

- Faster R-CNN: `fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth`,
  SHA256 `907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`.
- EasyOCR: `english_g2.pth`, SHA256
  `e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8`.
  Upstream MD5 `5864788e1821be9e454ec108d61b887d` также проверяется EasyOCR.

OCR берёт область туловища в исходном разрешении, отвергает мелкие/размытые
кропы, хранит несколько наблюдений; минимум три согласованных чтения с долей
0.8 лишь предлагают номер. Не выводит команду из одного цвета. Номер не
доказывает личность при одинаковых номерах команд. Краткосрочные траектории
Norfair могут иметь ID switches; это пока не измерено на волейболе. После
потери прежний игрок не назначается ближайшему человеку. Второй проход
создаёт гипотезы по номеру+внешнему виду+непересечению времени, без автоматического
принятия. Надёжное ReID, распознавание замен и событий пока не готовы.

В v0.3 для памяти внешности используется уже установленный COCO MobileNet/FPN
на изображении целого человека: 512 нормализованных признаков, версия по SHA.
Это общий детекторный backbone, не проверенная модель ReID. Сходство формирует
только гипотезу личности. Номера необязательны; appearance полностью отключает
OCR, `fetch-models --no-ocr` не загружает OCR-веса. Подробнее — IDENTITY.md.

Исходники и документация:

- https://pytorch.org/get-started/locally/
- https://pytorch.org/get-started/previous-versions/ — проверенная пара 2.13.0/0.28.0,
  Linux/Windows CPU, cu126, cu130, cu132; CUDA toolkit отдельно не устанавливается.
- https://github.com/pytorch/vision/blob/v0.28.0/LICENSE
- https://github.com/pytorch/vision#pre-trained-models
- https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html
- https://github.com/Megvii-BaseDetection/YOLOX
- https://www.ultralytics.com/license
- https://github.com/tryolabs/norfair
- https://github.com/JaidedAI/EasyOCR
- https://github.com/JaidedAI/EasyOCR/blob/v1.7.2/easyocr/config.py
- https://cocodataset.org/#termsofuse
- https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
- https://ffmpeg.org/ffprobe.html

FFmpeg/ffprobe запускаются локально. PyAV включает библиотеки FFmpeg; системная
сборка и резервный binary imageio-ffmpeg могут различаться набором кодеков и
условиями GPL/LGPL (libx264 обычно GPL). PySide6 — LGPLv3/GPLv3/commercial;
используется динамически из отдельного окружения. Переносимый Windows-пакет
включает Windows wheels и их исходные license/notice файлы; DLL остаются отдельно
заменяемыми. Источники и условия — packaging/windows/THIRD_PARTY.txt.
Перед распространением сборки учитывать условия
именно поставляемых бинарников. Код приложения не требует платных сервисов.
