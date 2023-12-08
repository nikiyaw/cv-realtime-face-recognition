import cv2
import numpy as np
from PIL import Image
import os
import glob

# Path for face image database
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

print('\n [INFO] Training faces. It will take a few seconds. Please wait ...')

faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save training model
recognizer.save('trainer/trainer.yml')

# Print the number of faces trained and end program
print('\n [INFO] {0} faces trained. Exiting Program'.format(len(np.unique(ids))))


# Note: cd facialRecognitionProject FIRST!!
# Note: cd dataset -> rm .ds_store -> cd .. -> run again