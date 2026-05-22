import os
import shutil

# Zip'ten çıkan ana klasörün yolu (Senin bilgisayarına göre güncelledik)
kaynak_dizin = "dataset/archive" 

# PyTorch'un okuyabilmesi için oluşturacağımız temiz klasörler
acik_hedef = "dataset/Acik"
kapali_hedef = "dataset/Kapali"

# Klasörleri oluştur
os.makedirs(acik_hedef, exist_ok=True)
os.makedirs(kapali_hedef, exist_ok=True)

sayac_acik = 0
sayac_kapali = 0

print("Resimler ayrıştırılıyor, lütfen bekleyin. Bu işlem 1-2 dakika sürebilir...")

# Tüm alt klasörleri gez ve resimleri bul
for kok_dizin, alt_dizinler, dosyalar in os.walk(kaynak_dizin):
    for dosya in dosyalar:
        if dosya.endswith(".png"):
            # Dosya adını alt çizgilere göre parçala
            parcalar = dosya.split('_')
            
            # İsmin uzunluğunu kontrol et (hatalı dosyaları atlamak için)
            if len(parcalar) > 4:
                goz_durumu = parcalar[4] # 5. sıradaki değer (index 4)
                kaynak_yol = os.path.join(kok_dizin, dosya)
                
                # 0 = Kapalı, 1 = Açık
                if goz_durumu == '0':
                    hedef_yol = os.path.join(kapali_hedef, f"kapali_{sayac_kapali}.png")
                    shutil.copy(kaynak_yol, hedef_yol)
                    sayac_kapali += 1
                elif goz_durumu == '1':
                    hedef_yol = os.path.join(acik_hedef, f"acik_{sayac_acik}.png")
                    shutil.copy(kaynak_yol, hedef_yol)
                    sayac_acik += 1

print(f"İşlem Tamamlandı! Toplam {sayac_acik} Açık, {sayac_kapali} Kapalı göz resmi dataset klasörüne kopyalandı.")