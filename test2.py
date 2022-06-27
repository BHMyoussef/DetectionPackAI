import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while True:
   ret, img = cap.read()
   img = cv2.resize(img,(340, 220))
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   panneaux = stop_cascade.detectMultiScale(gray, 1.3, 5)
   for (x, y, w, h) in panneaux:
       cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
       panneau = img[y:y + h, x:x + w]
       cv2.imshow('panneau STOP', panneau)

   cv2.imshow('img', img)
   key = cv2.waitKey(1) & 0xFF
   if key == ord("q"):
       break
cap.release()
cv2.destroyAllWindows()