import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)   #Kamerayı açıp görüntüyü yakalıyoruz.
cap.set(3, 640)   #Kamera Yüksekliği ayarlandı.
cap.set(4, 480)   #Kamera Genişliği ayarlandı.


mpHand = mp.solutions.hands   #mediapipe ile solutions modülünden ellerimizi alıyoruz.
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils   #Elimizin üzerindeki eklemler arasındaki bağlantıyı çizmemizi sağlar.


tipIds = [4, 8, 12, 16, 20]   #Parmaklarımızın uç noktalarının id lerini belirliyoruz.
while True:
    succes,img = cap.read()   #read() fonksiyonu 2 değer geriye döner, birinci paramerte olarak True yada False döner, ikinci parametre olarak Video kayıt anındaki her bir resim karesini döner.
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   #OpenCV renkleri BGR olarak algılar. Gerçek Hayatta RGB renkleri kullanıldığımız için BGR dan RGB ye renk dönüşümü gerçekleştridik.
    
    
    results = hands.process(imgRGB)   #Yukarıda çağırdığımız modülü kullanbilmek için "process" fonksiyonunu kullandık.
    #print(results.multi_hand_landmarks)   #Bir üst satırda çağırdığımız "process" fonksiyonu sayesinde elimizde bulunan eklemlerin kordinatlarının tespitini gerçekleştridik. Ve kordinat bilgilerini ekrana yazdırdık.
    
    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)   #Burada elimizdeki iskelet yapısını çizdiriyoruz.
            
            for id, lm in enumerate(handLms.landmark):   #Elimizdeki eklemlerin (x,y) kordinatlarını lm içerisine , kordinatların id'sini id içerisine aktarıyoruz.
                h, w, _ = img.shape   #shape fonksiyonu 3 şey return eder, 1=yükseklik 2=genişlik 3=renk kanalı , renk kanalıyla burada işimiz yok o yüzden boş bıraktık.
                cx, cy = int(lm.x*w), int(lm.y*h)   #Eklem noktalarının kordinatını bulmak için, x ekseni kordinatı ile genişliği çarpıyoruz. y ekseni kordinatı ile yüksekliği çarpıyoruz ve tam sayı formatına çeviriyoruz
                lmList.append([id, cx, cy])
                
                
    if len(lmList) != 0:   #id ve kordinat bilgilerini tutan liste doluysa aşşağıdaki işlemler gerçekleşsin. 
        fingers = []
        
        
        #sağ baş parmak 
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:   #baş parmağımızın uç noktasındaki eklemin x eksenindeki konumu hemen bir alt tarafındaki ekleme göre küçük ise parmağı bükülmüş kabul ediyoruz.
            fingers.append(0)   #baş parmak büküldüyse 0 ekliyoruz listeye. 
        else:
            fingers.append(1)   #baş parmak bükülmediyse 1 ekliyoruz listeye.
            
            
        
        
        
        #Geri Kalan 4 parmak 
        for id in range(1, 5):   #başparmak haricindeki diğer parmakların id numalarını aşşağıda çağırıcaz.   
            
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:   #1.den başlayıp 4.id ye kadar liste içerisine parmağın uç noktalarının kordinat bilgileri geliyor. bu kordinat bilgilerinin 2.indeksinin(yani y ekseni) kordinat bilgisi , kendisinden -2 değer altındaki kordinat bilgisinden küçükse(yani daha alt konumdaysa y eksenine göre) bu parmak bükülmüştür.
                fingers.append(1)   #eğer parmak bükülmediyse fingers listesine 1 ekliyoruz
            else:
                fingers.append(0)   #eğer parmak büküldüyse fingers listesine 0 ekliyoruz
           
        totalFinger = fingers.count(1)
        #print(totalFinger)   #parmakta bükülme durumunu gösterme.
        print(fingers)   #0 ve 1 lerden oluşan fingers listesini buradan görebilirsiniz.
    
        cv2.putText(img, str(totalFinger), (30, 130), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 8)   #parmak bükülme durumunun sayısını videoya yazan kod.
                
                
                
    
    cv2.imshow("Parmak Sayma Goruntusu",img)   #Görüntüyü açma
    cv2.waitKey(1)   #Belirli bir süre(1 mili saniye) bekletip kodun devam etmesini sağlar.
    
    
    
       
    