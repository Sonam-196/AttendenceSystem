# app.py
import os
import io
import cv2
import time
import math
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

# Patch for older insightface code that uses np.int
import numpy as _np
if not hasattr(_np, 'int'):
    _np.int = int

# --------- ML / DL imports -----------
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from insightface.app import FaceAnalysis

# --------- CONFIG ----------
ENROLLED_DIR = "enrolled_faces"         # raw uploaded enrollment images
ALIGNED_DIR = "aligned_enrolled"        # created automatically
EMBED_FILE = "face_embeddings.npy"
CLASSIFIER_FILE = "resnet18_classifier.pth"
MLP_FILE = "mlp_on_embeddings.pth"
ATTENDANCE_CSV = "attendance_log.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------- Utility functions ----------
def ensure_dirs():
    os.makedirs(ENROLLED_DIR, exist_ok=True)
    os.makedirs(ALIGNED_DIR, exist_ok=True)

# parse name & enrollment from filename: Name_Enrollment.ext
def parse_name_enroll(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    name = parts[0]
    enroll = ""
    if len(parts) > 1:
        # join anything numeric as enrollment
        digits = [p for p in parts[1:] if any(ch.isdigit() for ch in p)]
        enroll = digits[0] if digits else ""
    # fallback: try extract digits from name
    if enroll == "":
        import re
        m = re.search(r"(\d{6,})", base)
        if m:
            enroll = m.group(1)
    return name, enroll

# --------- InsightFace initialization ----------
@st.cache_resource(show_spinner=False)
def init_insightface():
    app = FaceAnalysis(name='buffalo_l')   # uses retinaface + arcface heavy model
    app.prepare(ctx_id=0, det_size=(640,640))
    # lower threshold a bit for robustness
    try:
        app.det_thresh = 0.25
    except Exception:
        pass
    return app

# Align enrolled images using InsightFace detector and save into ALIGNED_DIR
def align_enrolled_images(app):
    ensure_dirs()
    files = [f for f in os.listdir(ENROLLED_DIR) if f.lower().endswith((".jpg", ".jpeg"))]
    if not files:
        return {"status":"no_images"}
    saved = []
    for file in files:
        path = os.path.join(ENROLLED_DIR, file)
        img = cv2.imread(path)
        if img is None:
            continue
        # add slight padding and brightness to help detection if needed
        h,w = img.shape[:2]
        pad = int(0.15 * max(h,w))
        img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
        img_pad = cv2.convertScaleAbs(img_pad, alpha=1.2, beta=15)
        faces = app.get(img_pad)
        if len(faces) == 0:
            # try resizing larger and try again
            img2 = cv2.resize(img_pad, (800,800))
            faces = app.get(img2)
            working_img = img2
        else:
            working_img = img_pad

        if len(faces) == 0:
            # as last resort, save the padded image (so user can review)
            out = os.path.join(ALIGNED_DIR, file)
            cv2.imwrite(out, working_img)
            saved.append((file, False))
            continue

        # take largest face, add margin
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        f = faces[0]
        x1,y1,x2,y2 = f.bbox.astype(int)
        # add margin
        pad2 = int(0.2 * max(x2-x1, y2-y1))
        x1 = max(0, x1-pad2); y1 = max(0, y1-pad2)
        x2 = min(working_img.shape[1], x2+pad2); y2 = min(working_img.shape[0], y2+pad2)
        face_crop = working_img[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (320,320))
        out = os.path.join(ALIGNED_DIR, file)
        cv2.imwrite(out, face_crop)
        saved.append((file, True))
    return {"status":"done", "saved": saved}

# Extract ArcFace embeddings for all aligned enrolled images and save to EMBED_FILE
def extract_embeddings(app):
    files = [f for f in os.listdir(ALIGNED_DIR) if f.lower().endswith((".jpg",".jpeg"))]
    embeddings = {}
    for f in files:
        path = os.path.join(ALIGNED_DIR, f)
        img = cv2.imread(path)
        if img is None:
            continue
        faces = app.get(img)
        if len(faces) == 0:
            continue
        emb = faces[0].normed_embedding  # normalized embedding (512-d)
        embeddings[f] = emb
    np.save(EMBED_FILE, embeddings)
    return embeddings

# Simple dataset for training ResNet on aligned enrolled faces
class EnrolledDataset(Dataset):
    def __init__(self, folder, transform):
        self.samples = []
        self.transform = transform
        for f in os.listdir(folder):
            if not f.lower().endswith((".jpg",".jpeg")):
                continue
            name, enroll = parse_name_enroll(f)
            # label: use enrollment number if present else name
            label = enroll if enroll!="" else name
            self.samples.append((os.path.join(folder,f), label))
        # build label mapping
        labels = sorted(list({s[1] for s in self.samples}))
        self.label2idx = {l:i for i,l in enumerate(labels)}
        self.idx2label = {i:l for l,i in self.label2idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path,label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label2idx[label]

# Train ResNet18 classifier on aligned_enrolled (warning: needs multiple images per class to generalize)
def train_resnet_classifier(epochs=10, lr=1e-3, batch=8):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    ds = EnrolledDataset(ALIGNED_DIR, transform)
    if len(ds)==0:
        return {"status":"no_data"}
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    num_classes = len(ds.label2idx)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)
    # small training loop (fine-tuning from scratch is not ideal with few images)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        running = 0.0
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            opt.zero_grad()
            out = model(imgs)
            loss = lossfn(out, labels)
            loss.backward()
            opt.step()
            running += loss.item()
        # you can print ep loss to the streamlit log
    # save
    torch.save({"model_state": model.state_dict(), "label2idx": ds.label2idx}, CLASSIFIER_FILE)
    return {"status":"trained", "classes": ds.label2idx}

# Small MLP on ArcFace embeddings (optional)
class MLP(nn.Module):
    def __init__(self, in_dim=512, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self,x):
        return self.net(x)

def train_mlp_on_embeddings(epochs=30, lr=1e-3):
    # load embeddings and labels from aligned file names
    if not os.path.exists(EMBED_FILE):
        return {"status":"no_embeddings"}
    data = np.load(EMBED_FILE, allow_pickle=True).item()
    X=[]; y=[]
    labels=[]
    for fname, emb in data.items():
        name, enroll = parse_name_enroll(fname)
        label = enroll if enroll!="" else name
        labels.append(label)
    classes = sorted(list(set(labels)))
    label2idx = {l:i for i,l in enumerate(classes)}
    for fname, emb in data.items():
        name, enroll = parse_name_enroll(fname)
        label = enroll if enroll!="" else name
        X.append(emb)
        y.append(label2idx[label])
    X = torch.tensor(np.stack(X)).float()
    y = torch.tensor(y).long()
    num_classes = len(classes)
    mlp = MLP(in_dim=X.shape[1], num_classes=num_classes).to(DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    lossfn = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(X,y)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    for ep in range(epochs):
        mlp.train()
        for xb,yb in loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            out = mlp(xb)
            loss = lossfn(out, yb)
            loss.backward(); opt.step()
    torch.save({"model_state":mlp.state_dict(), "label2idx":label2idx}, MLP_FILE)
    return {"status":"trained_mlp"}

# Predict using combined logic: ArcFace cosine + ResNet classifier + optional MLP
def predict_and_mark(app, group_img_path, threshold=0.5, use_resnet=True, use_mlp=True):
    # load embeddings db
    db = {}
    if os.path.exists(EMBED_FILE):
        db = np.load(EMBED_FILE, allow_pickle=True).item()
    enrolled_names = list(db.keys())
    enrolled_embeds = np.array(list(db.values())) if db else np.array([])
    # load resnet classifier if exists
    resnet_model = None; resnet_labelmap = None
    if use_resnet and os.path.exists(CLASSIFIER_FILE):
        ckpt = torch.load(CLASSIFIER_FILE, map_location=DEVICE)
        label2idx = ckpt.get("label2idx", {})
        num_classes = len(label2idx)
        # instantiate model and load
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(ckpt["model_state"])
        model.to(DEVICE).eval()
        resnet_model = model
        resnet_labelmap = {v:k for k,v in label2idx.items()}

    mlp_model = None; mlp_labelmap = None
    if use_mlp and os.path.exists(MLP_FILE):
        ck = torch.load(MLP_FILE, map_location=DEVICE)
        # reconstruct mlp
        label2idx = ck.get("label2idx", {})
        num_classes = len(label2idx)
        mlp = MLP(in_dim=512, num_classes=num_classes)
        mlp.load_state_dict(ck["model_state"])
        mlp.to(DEVICE).eval()
        mlp_model = mlp
        mlp_labelmap = {v:k for k,v in label2idx.items()}

    # load group image
    img = cv2.imread(group_img_path)
    if img is None:
        raise FileNotFoundError(group_img_path)
    faces = app.get(img)
    results = []
    if len(faces) == 0:
        return {"status":"no_faces"}

    for face in faces:
        bbox = face.bbox.astype(int)
        emb = face.normed_embedding  # 512-d normalized
        best_name = "Unknown"; best_score = -1.0
        # arcface matching
        if enrolled_embeds.size:
            sims = cosine_similarity(emb.reshape(1,-1), enrolled_embeds)[0]
            idx = np.argmax(sims)
            best_score = float(sims[idx])
            candidate = enrolled_names[idx]
            best_name = candidate if best_score >= threshold else "Unknown"

        # resnet prediction (if available)
        resnet_choice = None; resnet_conf = 0.0
        if resnet_model is not None:
            # crop & preprocess same as training
            x1,y1,x2,y2 = bbox
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                resnet_choice=None
            else:
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                ])
                xin = transform(pil).unsqueeze(0).to(DEVICE)
                out = resnet_model(xin)
                probs = torch.softmax(out, dim=1).cpu().detach().numpy().flatten()
                idx = int(np.argmax(probs))
                resnet_conf = float(probs[idx])
                resnet_choice = resnet_labelmap.get(idx, None)

        # mlp choice
        mlp_choice = None; mlp_conf = 0.0
        if mlp_model is not None:
            xin = torch.tensor(emb).float().unsqueeze(0).to(DEVICE)
            out = mlp_model(xin)
            probs = torch.softmax(out, dim=1).cpu().detach().numpy().flatten()
            idx = int(np.argmax(probs))
            mlp_conf = float(probs[idx])
            mlp_choice = mlp_labelmap.get(idx, None)

        # combine signals: prefer ArcFace if score >= threshold,
        # else if resnet_conf high (>=0.6) prefer resnet_choice, else mlp if >=0.6.
        final_name = best_name
        final_conf = best_score
        if final_name == "Unknown":
            if resnet_choice and resnet_conf >= 0.6:
                final_name = resnet_choice
                final_conf = resnet_conf
            elif mlp_choice and mlp_conf >= 0.6:
                final_name = mlp_choice
                final_conf = mlp_conf

        results.append({
            "bbox": bbox.tolist(),
            "name": final_name,
            "score": float(final_conf),
            "arc_score": float(best_score),
            "resnet_choice": resnet_choice,
            "resnet_conf": resnet_conf,
            "mlp_choice": mlp_choice,
            "mlp_conf": mlp_conf
        })

    # create annotated image and attendance csv rows
    annotated = img.copy()
    now = datetime.now()
    rows = []
    for r in results:
        x1,y1,x2,y2 = r["bbox"]
        name = r["name"]
        score = r["score"]
        label = name
        color = (0,255,0) if name!="Unknown" else (0,0,255)
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        txt = f"{name} {score:.2f}"
        cv2.putText(annotated, txt, (x1, max(10,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # parse enrollment number from name if present (expects Name_Enrollment.jpg key)
        enroll = ""
        if name!="Unknown":
            parsed_name, parsed_enr = parse_name_enroll(name)
            enroll = parsed_enr
            # If the enrolled db key included the input filename, prefer that mapping:
            # if enrolled_db keys are filenames, we may want to map to display name
            display_name = parsed_name
        else:
            display_name = "Unknown"
        present = "Yes" if name!="Unknown" else "No"
        rows.append({
            "Name": display_name,
            "Enrollment": enroll,
            "Date": now.date().isoformat(),
            "Time": now.time().strftime("%H:%M:%S"),
            "Confidence": round(score,3),
            "Present": present
        })

    return {"status":"ok", "annotated": annotated, "rows": rows}

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Face Attendance (Major Project)", layout="wide")
st.title("📸 Face Recognition Attendance — Streamlit Demo")

st.sidebar.header("Setup / Actions")
app = init_insightface()

if st.sidebar.button("1) Align enrolled images (use enrolled_faces/)"):
    with st.spinner("Aligning enrolled images..."):
        res = align_enrolled_images(app)
    st.sidebar.write(res)

if st.sidebar.button("2) Extract ArcFace embeddings (from aligned_enrolled/)"):
    with st.spinner("Extracting embeddings..."):
        emb = extract_embeddings(app)
    st.sidebar.write({"embeddings_count": len(emb)})

if st.sidebar.button("3) Train ResNet18 classifier (aligned_enrolled/)"):
    with st.spinner("Training ResNet18 (quick)..."):
        tr = train_resnet_classifier(epochs=5)
    st.sidebar.write(tr)

if st.sidebar.button("4) Train MLP on embeddings (optional)"):
    with st.spinner("Training MLP on embeddings..."):
        tr2 = train_mlp_on_embeddings(epochs=30)
    st.sidebar.write(tr2)

st.sidebar.markdown("---")
st.sidebar.write("Make sure:")
st.sidebar.write("- Enrolled images placed in `enrolled_faces/` as JPG, named `Name_Enrollment.jpg`")
st.sidebar.write("- Then run 1) then 2) to create `face_embeddings.npy`")

st.header("Run Attendance on a Group Photo")
uploaded = st.file_uploader("Upload group photo (JPG)", type=["jpg","jpeg","png"])
threshold = st.slider("Matching threshold (ArcFace cosine)", 0.2, 0.9, 0.5, 0.05)
use_resnet = st.checkbox("Use ResNet classifier signal (if trained)", value=True)
use_mlp = st.checkbox("Use MLP-on-embeddings signal (if trained)", value=False)

if uploaded is not None:
    # save uploaded
    bytes_data = uploaded.read()
    img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    group_path = "group_photo.jpg"
    img.save(group_path)
    st.image(img, caption="Uploaded group photo", use_column_width=True)

    if st.button("Run attendance"):
        with st.spinner("Detecting, matching, and creating attendance... this may take a few seconds"):
            try:
                out = predict_and_mark(app, group_path, threshold=threshold, use_resnet=use_resnet, use_mlp=use_mlp)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                out = None
        if out is None:
            st.error("Prediction failed.")
        elif out.get("status") == "no_faces":
            st.warning("No faces detected in group photo.")
        elif out.get("status") == "ok":
            annotated = out["annotated"]
            rows = out["rows"]
            # save csv (append)
            df = pd.DataFrame(rows)
            if os.path.exists(ATTENDANCE_CSV):
                df_existing = pd.read_csv(ATTENDANCE_CSV)
                df = pd.concat([df_existing, df], ignore_index=True)
            df.to_csv(ATTENDANCE_CSV, index=False)
            # show annotated image
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Attendance Result", use_column_width=True)
            st.success(f"Attendance saved to {ATTENDANCE_CSV}")
            st.dataframe(df)
            # allow download
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download attendance CSV", data=csv_bytes, file_name="attendance.csv", mime="text/csv")
        else:
            st.write(out)
