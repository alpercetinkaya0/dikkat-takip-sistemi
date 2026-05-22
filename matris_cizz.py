import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. IŞIK İYİLEŞTİRME (CLAHE)
class ApplyCLAHE:
    def __call__(self, img):
        img_np = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_np)
        return Image.fromarray(img_clahe)

# 2. MODEL MİMARİSİ (v3)
class GozTakipModeli_v3(nn.Module):
    def __init__(self):
        super(GozTakipModeli_v3, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.50)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. VERİ HAZIRLIĞI
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    ApplyCLAHE(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root='dataset', transform=transform)
test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

# 4. MODELİ YÜKLE VE ANALİZ ET
model = GozTakipModeli_v3().to(device)
if os.path.exists('goz_modeli_v3.pth'):
    model.load_state_dict(torch.load('goz_modeli_v3.pth', map_location=device))
    model.eval()
    print("✅ Model yüklendi, matris çiziliyor...")
else:
    print("❌ HATA: goz_modeli_v3.pth bulunamadı!")
    exit()

gercekler = []
tahminler = []

with torch.no_grad():
    for veriler, etiketler in test_loader:
        veriler = veriler.to(device)
        ciktilar = model(veriler)
        _, preds = torch.max(ciktilar, 1)
        gercekler.extend(etiketler.cpu().numpy())
        tahminler.extend(preds.cpu().numpy())

# 5. GÖRSELLEŞTİRME
cm = confusion_matrix(gercekler, tahminler)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.title('Karmaşıklık Matrisi (Confusion Matrix)')
plt.ylabel('Gerçek Sınıf')
plt.xlabel('Model Tahmini')
plt.savefig('karmasiklik_matrisi.png')
plt.show()