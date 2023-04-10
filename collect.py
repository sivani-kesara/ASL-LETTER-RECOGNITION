import cv2
import numpy as np
import os
import string
alphabet = "abcdefghijklmnopqrstuvwxyz"
for subdir in ['train', 'test']:
    for letter in string.ascii_uppercase:
        path = f"data/{subdir}/{letter}"
        if not os.path.exists(path):
            os.makedirs(path)
mode = 'train'
directory = 'data/'+mode+'/'
mode_test = 'test'
directory_test = 'data/'+mode_test+'/'
max_images_per_letter = 1000
minValue = 100
cap = cv2.VideoCapture(0)
interrupt = -1  
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    count = {letter: len(os.listdir(directory+"/"+letter.upper())) for letter in alphabet}
    for i, letter in enumerate(count.keys()):
        cv2.putText(frame, f"{letter.upper()} : {count[letter]}", (10, 100+i*10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    cv2.rectangle(frame, (220-1, 9), (620+1, 419), (255,0,0) ,1)
    roi = frame[10:410, 220:520]
    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    test_image = cv2.resize(test_image, (300,300))
    cv2.imshow("test", test_image)
    interrupt = cv2.waitKey(10)
    for key in count.keys():
        if interrupt & 0xFF == ord(key):
            if count[key] < max_images_per_letter:
                cv2.imwrite(directory+key.upper()+'/'+str(count[key])+'.jpg', roi)
                cv2.imwrite(directory_test+key.upper()+'/'+str(count[key])+'.jpg', roi)
            else:
                print(f"Maximum images reached for letter {key.upper()}. Skipping capture.")
    if interrupt == 27:  # ASCII value for the escape key
        break
cap.release()
cv2.destroyAllWindows()