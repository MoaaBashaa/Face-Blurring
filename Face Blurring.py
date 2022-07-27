import cv2
import matplotlib.pyplot as plt # for plotting the image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('D:\FCAI\Dataset lfw\Adrian_McPherson\Adrian_McPherson_0001.jpg')

# Detect faces
faces = face_cascade.detectMultiScale(image = img, scaleFactor = 1.1, minNeighbors = 5)

# Draw Rec for the face/s
for (x, y, w, h) in faces:
      cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
      
img[y:y+h, x:x+w] = cv2.medianBlur(img[y:y+h, x:x+w], 49)


# Showing number of faces detected in the image
print(len(faces),"faces detected!")

# Plotting the image with face detected
finalimgcolor = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
finalimggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(12,12))
plt.imshow(finalimgcolor) 
plt.axis("off")
plt.show()
cv2.imshow('result',img)
cv2.waitKey(0)