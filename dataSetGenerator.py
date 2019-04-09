import cv2
import numpy as np
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('Classifiers/face.xml')
i=0
offset=50

name_person = input('Enter name of person : ')

with open("id_count.txt", "r") as file:
    name = file.read()

with open("id_count.txt", "w") as file:
    file.write(str(int(name)+1))

with open("hash_faces.txt", "a") as file:
    file.write(name + " " + name_person + "\n")
    
while True:
    ret, im =cam.read()
    im = np.array(im, dtype=np.uint8)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        i=i+1
        cv2.imwrite("dataSet/face-"+name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.waitKey(100)
    if i>100:
        cam.release()
        cv2.destroyAllWindows()
        break

