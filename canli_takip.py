import cv2                 
import torch               
import numpy as np         
from PIL import Image      
from torchvision import transforms  
import os                  
import time                
import winsound            

# 1. YENİ MİMARİNİN İSKELETİ (v3 - Eğitimdekiyle tamamen aynı)
class GozTakipModeli_v3(torch.nn.Module):
    def __init__(self):
        super(GozTakipModeli_v3, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64) 
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.50)
        self.fc1 = torch.nn.Linear(128 * 8 * 8, 512) 
        self.fc2 = torch.nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. YENİ MODELİ YÜKLEME
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GozTakipModeli_v3().to(device)

if os.path.exists('goz_modeli_v3.pth'):
    model.load_state_dict(torch.load('goz_modeli_v3.pth', map_location=device))
    model.eval()
    print("✅ Model v3 (Düşük Işık Dirençli) başarıyla yüklendi!")
else:
    print("❌ HATA: goz_modeli_v3.pth bulunamadı!")

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# O meşhur ışık iyileştirme (CLAHE) adımı
def preprocess(eye_roi):
    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    img = Image.fromarray(gray)
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    return transform(img).unsqueeze(0).to(device)

# --- AYARLAR ---
kapali_baslangic_suresi = None 
UYARI_ESIGI = 5.0  # Gözler 5 saniye kapalı kalırsa alarm çalar

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_frame, 1.1, 7) 

    acik_goz_sayisi = 0

    if len(eyes) >= 2:
        for (x, y, w, h) in eyes:
            eye_roi = frame[y:y+h, x:x+w]
            if eye_roi.size > 0:
                input_tensor = preprocess(eye_roi)
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.softmax(output, dim=1)
                    if prob[0][0].item() > 0.90:
                        acik_goz_sayisi += 1
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # ODAK VE ZAMAN KONTROLÜ
    if acik_goz_sayisi < 2:
        if kapali_baslangic_suresi is None:
            kapali_baslangic_suresi = time.time()
        
        gecen_sure = time.time() - kapali_baslangic_suresi
        
        if gecen_sure > UYARI_ESIGI:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 15)
            cv2.putText(frame, f"ODAKLAN! ({int(gecen_sure)}s)", (150, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            winsound.Beep(1000, 200) 
    else:
        kapali_baslangic_suresi = None
        cv2.putText(frame, "SISTEM AKTIF: ODAK TAM", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Odak Takibi v3', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()