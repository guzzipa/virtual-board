import cv2
import mediapipe as mp
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=float(self.detectionCon),
            min_tracking_confidence=float(self.trackCon)
        )
        self.mpDraw = mp.solutions.drawing_utils  # objetos para dibujar
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self,frame,draw=True):
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)#convierto a RGB
        self.results=self.hands.process(frameRGB)#proceso RGB
        if self.results.multi_hand_landmarks:# x,y,z de cada landmark o NONE (si no hay manos)
            for handLms in self.results.multi_hand_landmarks:#cada marca de mano
                if draw:
                     self.mpDraw.draw_landmarks(frame,handLms,self.mpHands.HAND_CONNECTIONS)#uniendo puntos de nuestra mano!
        
        return frame

    def findPosition(self,frame,handNo=0,draw=True):
        xList=[]
        yList=[]
        bbox=[]
        self.lmpoints=[]
        if self.results.multi_hand_landmarks:#x,y,z de cada marca
            myHand=self.results.multi_hand_landmarks[handNo]#resultados de mano particular
            for id,lm in enumerate(myHand.landmark):#id y lm(x,y,z)
                h,w,c=frame.shape#h,w for converting decimals x,y into pixels 
                cx,cy=int(lm.x*w),int(lm.y*h)# pixels coordinates for landmarks
                # print(id, cx, cy)
                xList.append(cx)
                yList.append(cy)
                self.lmpoints.append([id,cx,cy])
                if draw:
                    cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)    
            xmin,xmax=min(xList),max(xList)
            ymin,ymax=min(yList),max(yList)
            bbox=xmin,ymin,xmax,ymax

            if draw:
                cv2.rectangle(frame,(bbox[0]-20,bbox[1]-20),(bbox[2]+20,bbox[3]+20),(0,255,0),2)

        return self.lmpoints,bbox

    def fingersUp(self):#dedo arriba ¿cual?
        fingers = []
        if self.lmpoints[self.tipIds[0]][1] > self.lmpoints[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        for id in range(1, 5):
            if self.lmpoints[self.tipIds[id]][2] < self.lmpoints[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, frame, draw=True,r=15,t=3):# distancia entre dos puntos
        x1, y1 = self.lmpoints[p1][1],self.lmpoints[p1][2]#x,y de p1
        x2, y2 = self.lmpoints[p2][1],self.lmpoints[p2][2]#x,y de p1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2#punto promedio

        if draw: #dibujando lineas 
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, cx, cy]
    

def main():
    cap=cv2.VideoCapture(0)
    detector=handDetector()

    while True:
        capture,frame=cap.read()#
        frame =detector.findHands(frame)
        lmpoints,bbox= detector.findPosition(frame)
        if len(lmpoints)!=0:
            print(lmpoints[4])
            
        cv2.imshow("Image",frame)
        cv2.waitKey(1)


if __name__=="__main__":
    main()