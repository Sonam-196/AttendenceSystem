from insightface.app import FaceAnalysis
import os, cv2, numpy as np

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640,640))
app.det_thresh = 0.25

embeddings = {}
folder = "aligned_enrolled"

for file in os.listdir(folder):
    path = os.path.join(folder, file)
    img = cv2.imread(path)
    faces = app.get(img)
    if len(faces) == 0:
        print(f"❌ No face in {file}")
        continue
    emb = faces[0].embedding
    embeddings[file] = emb
    print(f"✅ Extracted embedding for {file}")

np.save("face_embeddings.npy", embeddings)
print("🎉 Saved all embeddings using ArcFace model.")
