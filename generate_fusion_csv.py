import os
import csv
from app import predict_and_mark, init_insightface

app = init_insightface()

INPUT_FOLDER = "fusion_samples"
OUTPUT_CSV = "fusion_dataset.csv"

with open(OUTPUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "arc", "resnet", "mlp", "label"])

    for img_name in os.listdir(INPUT_FOLDER):
        img_path = os.path.join(INPUT_FOLDER, img_name)

        out = predict_and_mark(app, img_path)

        print("DEBUG OUTPUT:", out)   # see structure

        # no face or unexpected output → skip safely
        if (
            out is None or 
            "results" not in out or 
            len(out["results"]) == 0
        ):
            w.writerow([img_name, "NA", "NA", "NA", "NA"])
            continue

        r = out["results"][0]

        arc = r.get("arc_score", "NA")
        res = r.get("resnet_conf", "NA")
        mlp = r.get("mlp_conf", "NA")
        label = r.get("name", "Unknown")

        w.writerow([img_name, arc, res, mlp, label])
        print("✔ Added:", img_name)
