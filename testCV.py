import cv2
import numpy as np
import face_recognition
# read image
imgElon = face_recognition.load_image_file('testImages/leonardo.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgElontest = face_recognition.load_image_file('testImages/Samuel Jackson.jpg')
imgElontest = cv2.cvtColor(imgElontest,cv2.COLOR_BGR2RGB)

# face location
faceloc=face_recognition.face_locations(imgElon)[0]
faceencode=face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest=face_recognition.face_locations(imgElontest)[0]
faceencodetest=face_recognition.face_encodings(imgElontest)[0]
cv2.rectangle(imgElontest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

# compare
results = face_recognition.compare_faces([faceencode],faceencodetest)
faceDis = face_recognition.face_distance([faceencode],faceencodetest)
print(results,faceDis)

# show image
cv2.imshow('Elon musk',imgElon)
cv2.imshow('Elon musk test',imgElontest)
cv2.waitKey(0)
