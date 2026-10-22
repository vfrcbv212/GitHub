"""OpenCV image encoding with Unicode-safe filesystem access on Windows."""
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    try:
        encoded=np.frombuffer(Path(path).read_bytes(),np.uint8)
        return cv2.imdecode(encoded,cv2.IMREAD_COLOR) if encoded.size else None
    except (OSError,cv2.error):return None


def write_image(path,image):
    path=Path(path)
    try:
        okay,encoded=cv2.imencode(path.suffix or '.jpg',image)
        if not okay:return False
        path.write_bytes(encoded.tobytes())
        return True
    except (OSError,cv2.error):return False
