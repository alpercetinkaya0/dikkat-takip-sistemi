import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. IŞIK İYİLEŞTİRME SINIFI
class ApplyCLAHE:
    def __call__(self, img):
        img_np = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_np)
        return Image.fromarray(img_clahe)

# 2. DONANIM VE VERİ ÖN İŞLEME
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    ApplyCLAHE(),             
    transforms.Resize((32, 32)), 
    transforms.ColorJitter(brightness=0.4, contrast=0.4), 
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root='dataset', transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 3. YENİ MİMARİ (v3 - Düşük Işık Dirençli)
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

model = GozTakipModeli_v3().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 4. EĞİTİM VE GRAFİK ÇİZİMİ
if __name__ == '__main__':
    epochs = 15 
    print(f"Eğitim başlıyor, Toplam Epoch: {epochs}")
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(epochs):
        model.train()
        train_hata = 0.0
        train_dogru = 0
        train_toplam = 0
        
        for veriler, etiketler in train_loader:
            veriler, etiketler = veriler.to(device), etiketler.to(device)
            optimizer.zero_grad()
            ciktilar = model(veriler)
            hata = criterion(ciktilar, etiketler)
            hata.backward()
            optimizer.step()
            
            train_hata += hata.item() * veriler.size(0)
            _, tahminler = torch.max(ciktilar, 1)
            train_dogru += (tahminler == etiketler).sum().item()
            train_toplam += etiketler.size(0)
            
        model.eval() 
        val_hata = 0.0
        val_dogru = 0
        val_toplam = 0
        
        with torch.no_grad(): 
            for veriler, etiketler in test_loader:
                veriler, etiketler = veriler.to(device), etiketler.to(device)
                ciktilar = model(veriler)
                hata = criterion(ciktilar, etiketler)
                
                val_hata += hata.item() * veriler.size(0)
                _, tahminler = torch.max(ciktilar, 1)
                val_dogru += (tahminler == etiketler).sum().item()
                val_toplam += etiketler.size(0)
                
        epoch_train_loss = train_hata / train_toplam
        epoch_val_loss = val_hata / val_toplam
        epoch_train_acc = train_dogru / train_toplam
        epoch_val_acc = val_dogru / val_toplam
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Kayıp: {epoch_train_loss:.4f}, Doğruluk: %{epoch_train_acc*100:.1f} | Val Kayıp: {epoch_val_loss:.4f}, Doğruluk: %{epoch_val_acc*100:.1f}")
        
    torch.save(model.state_dict(), 'goz_modeli_v3.pth')
    print("\n✅ Yeni Model Kaydedildi: goz_modeli_v3.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Eğitim (Train Acc)', marker='o')
    plt.plot(val_accuracies, label='Doğrulama (Val Acc)', marker='s')
    plt.title('Doğruluk Grafiği (Accuracy)')
    plt.legend()
    plt.savefig('dogruluk_grafigi.png')
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Eğitim (Train Loss)', marker='o')
    plt.plot(val_losses, label='Doğrulama (Val Loss)', marker='s')
    plt.title('Kayıp Grafiği (Loss)')
    plt.legend()
    plt.savefig('kayip_grafigi.png')
    plt.show()