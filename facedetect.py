import cv2
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
import os

# ---------- Initialize RetinaFace (InsightFace) ----------
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# ---------- Load image ----------
img_path = "group_photo.jpg"      # put your image name here
img = cv2.imread(img_path)

# ---------- Detect faces ----------
faces = app.get(img)
print(f"✅ Total faces detected: {len(faces)}")

# ---------- Draw boxes ----------
for i, face in enumerate(faces):
    box = face.bbox.astype(int)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.putText(img, f"Face {i+1}", (box[0], box[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# ---------- Save cropped faces ----------
os.makedirs("cropped_faces", exist_ok=True)
for i, face in enumerate(faces):
    box = face.bbox.astype(int)
    x1, y1, x2, y2 = box
    face_crop = img[y1:y2, x1:x2]
    cv2.imwrite(f"cropped_faces/face_{i+1}.jpg", face_crop)

print("✅ Cropped faces saved in 'cropped_faces' folder")

# ---------- Display ----------
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.axis("off")
plt.show()
