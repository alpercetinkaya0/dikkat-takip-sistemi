import cv2

# OpenCV'nin içinde hazır gelen Yüz ve Göz tanıma modellerini (Haar Cascades) yüklüyoruz
yuz_modeli = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
goz_modeli = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

kamera = cv2.VideoCapture(0)

while True:
    ret, frame = kamera.read()
    if not ret:
        print("Kameradan görüntü alınamadı!")
        break

    # 1. İşlemleri hızlandırmak için görüntüyü gri tonlamaya (siyah-beyaz) çeviriyoruz
    gri_kare = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Gri görüntü üzerinde yüzleri tespit ediyoruz
    yuzler = yuz_modeli.detectMultiScale(gri_kare, scaleFactor=1.3, minNeighbors=5)

    # 3. Bulunan her bir yüz için döngüye giriyoruz
    for (x, y, w, h) in yuzler:
        # Yüzün etrafına yeşil bir dikdörtgen çiz
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Sadece yüzün olduğu alanı "İlgi Alanı (ROI)" olarak kesiyoruz
        # Gözleri tüm ekranda değil, sadece yüzün içinde aramak daha hızlı ve doğru sonuç verir
        gri_yuz = gri_kare[y:y+h, x:x+w]
        renkli_yuz = frame[y:y+h, x:x+w]
        
        # 4. Sadece kırpılmış yüz bölgesi içinde gözleri arıyoruz
        gozler = goz_modeli.detectMultiScale(gri_yuz, scaleFactor=1.1, minNeighbors=10)
        
        # 5. Bulunan gözlerin etrafına sarı renkli bir dikdörtgen çiz
        for (gx, gy, gw, gh) in gozler:
            # OpenCV'de renkler BGR (Mavi, Yeşil, Kırmızı) sırasıyla yazılır.
            # (0, 255, 255) bize o canlı sarı rengi verecek.
            cv2.rectangle(renkli_yuz, (gx, gy), (gx+gw, gy+gh), (0, 255, 255), 2)

    # Sonucu ekranda göster
    cv2.imshow('Yuz ve Goz Tespiti', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()