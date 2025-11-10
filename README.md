# RetinaFace (local project)

This repository contains local code for a face-detection/face-embedding project (RetinaFace-based). Files include scripts for detection, embedding extraction, matching, and a simple app.

Files of note:
- `app.py` - main application
- `extract_embeddings.py` - extract face embeddings
- `facedetect.py` - face detection utilities
- `matches.py` - matching logic
- `aligned_enrolled/`, `enrolled_faces/` - image folders

Note: Large binary files (for example `face_embeddings.npy`) are ignored by default. If you want to include binaries, remove them from `.gitignore` or use Git LFS.

How to push to GitHub:
1. Create a repo on GitHub (for example `RetinaFace`).
2. Add the remote and push:
   ```powershell
   git remote add origin https://github.com/Sonam-196/<repo>.git
   git push -u origin main
   ```

Or use GitHub CLI:
```powershell
gh auth login
gh repo create Sonam-196/<repo> --public --source=. --remote=origin --push
```

License: add a license file if you want to set terms.
