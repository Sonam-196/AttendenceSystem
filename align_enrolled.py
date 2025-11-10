import cv2
from insightface.app import FaceAnalysis
import os

import numpy as np
if not hasattr(np, 'int'):
    np.int = int


input_dir = "enrolled_faces"
output_dir = "aligned_enrolled"
os.makedirs(output_dir, exist_ok=True)

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(".jpg"):
        continue

    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)

    # 💡 Preprocessing: add border + contrast
    img = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=[0,0,0])
    img = cv2.convertScaleAbs(img, alpha=1.3, beta=30)  # enhance brightness and contrast

    faces = app.get(img)
    print(f"{filename}: {len(faces)} faces detected")

    if len(faces) == 0:
        print(f"⚠️ No face found even after enhancement in {filename}")
        continue

    for i, face in enumerate(faces):
        aligned = app.draw_on(img, [face])
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, aligned)
        print(f"✅ Saved aligned face to {save_path}")
