import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

CSV = "fusion_dataset.csv"
MODEL = "fusion_model.pth"

df = pd.read_csv(CSV)

# remove NA rows
df = df[df.arc != "NA"]

X = df[["arc", "resnet", "mlp"]].astype(float).values
y = df["label"].values

enc = LabelEncoder()
y = enc.fit_transform(y)

X = torch.tensor(X).float()
y = torch.tensor(y).long()

ds = TensorDataset(X, y)
loader = DataLoader(ds, batch_size=8, shuffle=True)

class FusionNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = FusionNet(len(enc.classes_))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossfn = nn.CrossEntropyLoss()

for epoch in range(50):
    for xb, yb in loader:
        opt.zero_grad()
        out = model(xb)
        loss = lossfn(out, yb)
        loss.backward()
        opt.step()

torch.save({
    "model": model.state_dict(),
    "label_encoder": enc
}, MODEL)

print("Training done. Model saved →", MODEL)
