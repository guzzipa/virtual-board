import os

#path='/Users/pabloguzzi/Projects/virtual-board/'
#os.chdir(path)


import cv2
import handtracking as htm
import numpy as np

overImages=[]#lista de imagenes overlapeadas

brushThickness = 25
eraserThickness = 100
drawColor=(255,255,255)#color default

xp, yp = 0, 0
canvas = np.zeros((720, 1280, 3), np.uint8)

headerFolder="header"
myList=os.listdir(headerFolder)
#print(myList)

for imPath in myList:
    image=cv2.imread(f'{headerFolder}/{imPath}')
    resized_up = cv2.resize(image, (1280, 125), interpolation= cv2.INTER_LINEAR)
    overImages.append(resized_up)#inserting images one by one in the overImages

header=overImages[0]#storing 1st image 
cap=cv2.VideoCapture(1)
cap.set(3,1280)#width
cap.set(4,720)#height

detector = htm.handDetector(detectionCon=0.50,maxHands=1)
writer = None
file_out='output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID') 

while True:
    # Leo camara
    capture, frame = cap.read()
    frame=cv2.flip(frame,1)
    
    # detecto manos e imprimimos los landmarks
    frame = detector.findHands(frame)
    lmpoints,bbox = detector.findPosition(frame, draw=False)#funcion para entontrar position especifica, draw=false -> no dibujar cuadro detect mano
    
    if len(lmpoints)!=0:
        #print(lmpoints)
        x1, y1 = lmpoints[8][1],lmpoints[8][2]# dedo indice
        x2, y2 = lmpoints[12][1],lmpoints[12][2]# dedo medio
        
        #check de dedos estirados
        fingers = detector.fingersUp()

        # Modo seleccion -> dos dedos estirados ( indice y medio)
        #PEndiente relativizar las posiciones basado en la cantidad de funciones que se puedan hacer
        #Pendiente 2, parametrizar cantidad de funciones, sub-funciones dentro si las hay, y seteo de posicion de dedos para el uso de las mismas
        #parametros x izq, x der, image, color
        functions={}
        functions["black"]=[250,450,overImages[0], (255, 255, 255)]
        functions["red"]=[550,750,overImages[1], (253, 106, 63)]
        functions["blue"]=[800,950,overImages[2], (103, 103, 251)]
        functions["goma"]=[1050,1200,overImages[3], (0, 0, 0)]
        
        if fingers[1] and fingers[2]:
            xp,yp=0,0
            for i in functions.keys():
                if functions[i][0] < x1 < functions[i][1]:
                    header = functions[i][2]
                    drawColor = functions[i][3]
            cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)#modo seleccion representado por un cuadrado

        # Modo dibujo si el dedo indice esta estirado y el medio no
        if fingers[1] and fingers[2] == False:
            cv2.circle(frame, (x1, y1), 15, drawColor, cv2.FILLED)#reprensentado por un circulo (parametrizar el tamaño)
            #print("Drawing Mode")
            if xp == 0 and yp == 0:#inicializalizo en 0,0
                xp, yp = x1, y1
            #dibujamos, pero se elimina cada vez que se actualizan los frames, tenemos que definir el lienzo
            
            #borrar
            if drawColor == (0, 0, 0):
                cv2.line(frame, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(canvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(frame, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(canvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp,yp=x1,y1 # giving values to xp,yp everytime 
           
           #mergeamos frames
    
    # igm to gray
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    
    # convirtiendo en binario (radicalizo gama de grises) -> Creacion de mascara
    _, frameInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    frameInv = cv2.cvtColor(frameInv,cv2.COLOR_GRAY2BGR)#gris to RGB 
    
    #agregamos a original frame frameInv ( dibujo) solo negro
    frame = cv2.bitwise_and(frame,frameInv)
    
    #agreamos a imagen el canvas, ahora si impactan colores
    frame = cv2.bitwise_or(frame,canvas)
    
    #seteo header de seleccion
    frame[0:125,0:1280]=header
    
    cv2.imshow("Image", frame)
    #cv2.imshow("Canvas", canvas)
    #cv2.imshow("Inv", frameInv)
    
    #paso de escritura de video
    if writer is None and file_out is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(file_out, fourcc, 27.58,
                                 (frame.shape[1], frame.shape[0]), True)
        
    if writer is not None:
        writer.write(frame)
        
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    
    

# Release camara & destroy the windows.    
cap.release()
cv2.destroyAllWindows()
        


#Pasamos a mp4, avi pesa mucho!
#pendiente escribir mp4 directo ( se puede? )
from moviepy.editor import *
videoclipin = VideoFileClip(file_out)
videoclipout = VideoFileClip(file_out.replace('.avi','.mp4'))
audioclip = videoclipin.audio
videoclipout=videoclipout.fl_time(lambda t: t*0.997, keep_duration=False)
videoclipout=videoclipout.set_duration(audioclip.duration)
videoclipout=videoclipout.set_audio(audioclip)
videoclipout.write_videofile(file_out.replace('.avi','.mp4'))

