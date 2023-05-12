import os
import time
import xml.etree.ElementTree as ET

import cv2
from PIL import Image 
from imutils.video import VideoStream


class Detector:
    
    def __init__(self) -> None:
        self.camera_working = False
        self.names = []

    def enable_camera(self):
        
        if self.camera_working == False:
            self.vs = VideoStream(framerate=10,resolution=(300,300)).start()
            self.camera_working = True
        else :
            print('camera is already open')
        

    def start_regognition_once(self,face_cascade,recognizer):
            

        frame = self.vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h,x:x+w]
            _id,confidence = recognizer.predict(roi_gray)
            confidence = 100 - int(confidence)
            
            print(_id,confidence)

            if confidence > 50:
                text = self.names[_id]
                font = cv2.FONT_HERSHEY_PLAIN
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

            else:   
                text = "UnknownFace"
                font = cv2.FONT_HERSHEY_PLAIN
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)

        return frame


            


 


    def start_regognition_all(self,labels:list ):
            
            face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(f"./data/classifiers/trainner.xml")
       
            pred = 0
            while True:
                frame = self.vs.read()
                #default_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,1.3,5)

                for (x,y,w,h) in faces:


                    roi_gray = gray[y:y+h,x:x+w]

                    id,confidence = recognizer.predict(roi_gray)
                    confidence = 100 - int(confidence)
                    pred = 0
                    if confidence > 50:
                        #if u want to print confidence level
                                #confidence = 100 - int(confidence)
                                pred += +1
                                text = labels[id]
                                font = cv2.FONT_HERSHEY_PLAIN
                                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

                    else:   
                                pred += -1
                                text = "UnknownFace"
                                font = cv2.FONT_HERSHEY_PLAIN
                                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)

                # cv2.imshow("image", frame)
                
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame = jpeg.tobytes()
                
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break



              


            