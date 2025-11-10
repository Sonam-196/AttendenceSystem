import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------
# Safety patch for newer numpy versions (avoids np.int error)
if not hasattr(np, 'int'):
    np.int = int
# ------------------------------------------------------------------

# --- PATHS ---
enrolled_dir = "aligned_enrolled"       # enrolled faces after alignment
embeddings_file = "face_embeddings.npy" # output from extract_embeddings.py
group_photo_path = "group_photo.jpg"    # group photo input
output_path = "attendance_result.jpg"   # output with boxes and names

# --- Load ArcFace model ---
print("\n🚀 Loading InsightFace model ...")
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# --- Load stored embeddings ---
if not os.path.exists(embeddings_file):
    print(f"❌ '{embeddings_file}' not found! Run extract_embeddings.py first.")
    exit()

data = np.load(embeddings_file, allow_pickle=True).item()
enrolled_names = list(data.keys())
enrolled_embeds = np.array(list(data.values()))
print(f"✅ Loaded {len(enrolled_names)} enrolled faces from '{embeddings_file}'")

# --- Load group photo ---
if not os.path.exists(group_photo_path):
    print(f"❌ '{group_photo_path}' not found! Place a group image in this path.")
    exit()

img = cv2.imread(group_photo_path)
if img is None:
    print("❌ Could not read group photo.")
    exit()

print("📸 Detecting faces in group photo...")
faces = app.get(img)
print(f"🔍 Detected {len(faces)} faces in group photo.\n")

if len(faces) == 0:
    print("⚠️ No faces detected. Try a clearer photo.")
    exit()

# --- Matching Threshold ---
threshold = 0.5  # similarity threshold (you can adjust between 0.4–0.6)

# --- Process each detected face ---
for face in faces:
    bbox = face.bbox.astype(int)
    embedding = face.normed_embedding.reshape(1, -1)

    sims = cosine_similarity(embedding, enrolled_embeds)[0]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    name = enrolled_names[best_idx] if best_score >= threshold else "Unknown"

    print(f"🧠 Face matched: {name} (similarity={best_score:.2f})")

    # --- Draw bounding box + label ---
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"{name} ({best_score:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# --- Save result image ---
cv2.imwrite(output_path, img)
print(f"\n✅ Attendance result saved as '{output_path}'")

# --- Optionally show the image (press any key to close) ---
cv2.imshow("Attendance Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
