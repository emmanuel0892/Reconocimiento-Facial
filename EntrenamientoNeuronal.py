import cv2
import os 
import numpy as np
from PIL import Image

datos = 'RUTA DE LA CARPETA datosP'
personas = os.listdir(datos)
print('Lista de personas: ', personas)

labels = []
facesData = []
label = 0

for nameDir in personas:
    datosP = datos + '/' + nameDir
    print('Leyendo las imagenes')

    for fileName in os.listdir(datosP):
        print('Rostros: ', nameDir +'/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(datosP+'/'+fileName,0))
        image = cv2.imread(datosP+'/'+fileName,0)
        #cv2.imshow('image',image)
        #cv2.waitKey(10)
    label = label + 1

face_recognizer = cv2.face_LBPHFaceRecognizer.create()

print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

face_recognizer.write('modeloEigenFace.xml')

print("modelo almacenado")