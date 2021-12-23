import os
import cv2
import dlib

import numpy as nmpy
from keras.models import load_model
import time
from pygame import mixer
from datetime import datetime

FP = "....\\Driver Drowsiness files\\Face_predict.dat"
LMP = dlib.shape_predictor(FP)

Dtct = dlib.get_frontal_face_detector()
mixer.init()
frontal_face = cv2.CascadeClassifier('....\\Driver Drowsiness files\\Haar_cascade\HCC_frontalface.xml')
lefte = cv2.CascadeClassifier('....\\Driver Drowsiness files\\Haar_cascade\HCC_lefteye.xml')
righte = cv2.CascadeClassifier('....\\Driver Drowsiness files\\Haar_cascade\HCC_righteye.xml')
EyeAlert = mixer.Sound('....\\Driver Drowsiness files\\alarm.wav')
YawnAlert = mixer.Sound('....\\Driver Drowsiness files\\yawn_beep.wav')
Neural_net_model = load_model('....\\Driver Drowsiness files\\Neural_Net_Model.h5')

label=['Eyes_Closed','Eyes_Opened']

capture = cv2.VideoCapture(0)
cnt=0
thickness_border=2
txtfont = cv2.FONT_HERSHEY_DUPLEX
left_predictor=[99]
right_predictor=[99]
Dirp = os.getcwd()
Blink_rate=0
RoY = 0
RoY_chk = 10
Current_YS= False
Current_tme = datetime.now()



def MarkFacem(imge, facem):
    imge = imge.copy()
    for index, pnt in enumerate(facem):
        pt = (pnt[0, 0], pnt[0, 1])
        cv2.putText(imge, str(index), pt,txtfont,0.4,(0, 0, 255))
        cv2.circle(imge, pt, 3, (0, 255, 255))
    return imge

def botmlp(facem):
    botmlp_pt = []
    for i in range(65,68):
        botmlp_pt.append(facem[i])
    for i in range(56,59):
        botmlp_pt.append(facem[i])
    botmlp_avg = nmpy.mean(botmlp_pt,0) #axis=0
    return int(botmlp_avg[:,1])
        
def Facemarks(imge):
    rt = Dtct(imge, 1)

    if(len(rt) > 1):
        return "Warning"
    if(len(rt) == 0):
        return "Warning"
    return nmpy.matrix([[pred.x, pred.y] for pred in LMP(imge, rt[0]).parts()])   

def uprlp(facem):
    uprl_pt = []
    for i in range(50,53):
        uprl_pt.append(facem[i])
    for i in range(61,64):
        uprl_pt.append(facem[i])
    uprl_avg = nmpy.mean(uprl_pt,0) #axis=0
    return int(uprl_avg[:,1])    


def mth_opn(img):
    facem = Facemarks(img)
    
    if(facem == "Warning"):
        return img, 0
    
    facemarks_img = MarkFacem(img, facem)
    uprl_ctr = uprlp(facem)
    botmlp_ctr = botmlp(facem)
    Distlp = abs(uprl_ctr - botmlp_ctr)
    return facemarks_img, Distlp 
    
while True:
    ret, Img_frame = capture.read()   
    image_landmarks, Distlp = mth_opn(Img_frame)
    
    hgt,wd = Img_frame.shape[:2] 
    gry = cv2.cvtColor(Img_frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(Img_frame, (0,hgt-45) , (700,hgt) , (45,28,225) , cv2.FILLED )
    cv2.rectangle(Img_frame, (0,hgt-80) , (350,hgt-45) , (0,0,0) , cv2.FILLED )
    rightframes =  righte.detectMultiScale(gry)
    fframes = frontal_face.detectMultiScale(gry,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    leftframes = lefte.detectMultiScale(gry)
    
    First_YS = Current_YS  
    
    if (Distlp > 35):
        Current_YS = True 
        cv2.putText(Img_frame, "You Are Yawning!", (10,hgt-55), txtfont, 1,(0,0,255),2)

        cv2.putText(Img_frame," Rate Of Yawn: " + str(RoY + 1), (50,50),txtfont, 1,(0,0,255),2)
        if (RoY>RoY_chk):
            RoY_chk=RoY_chk+10
        
    else:
        Current_YS = False 
         
    if First_YS == True and Current_YS == False:
        RoY += 1
        
    if(RoY!=RoY_chk):
                Current_tme = datetime.now()
    else:
        if (datetime.now() - Current_tme).total_seconds() < 6:
                    cv2.putText(Img_frame,'Take a Break!',(100,200), txtfont, 2,(0,0,255),3,cv2.LINE_AA)
                    try:
                        YawnAlert.play()
                    except: 
                           pass
    
    for (xaxs,yaxs,wdh,ht) in leftframes:
        LeftEye=Img_frame[yaxs:yaxs+ht,xaxs:xaxs+wdh]
        cnt=cnt+1
        LeftEye = cv2.cvtColor(LeftEye,cv2.COLOR_BGR2GRAY)  
        LeftEye = cv2.resize(LeftEye,(24,24))
        left_eyee= LeftEye/255
        LeftEye=LeftEye.reshape(24,24,-1)
        LeftEye = nmpy.expand_dims(LeftEye,axis=0)
        predict_l_eye = Neural_net_model.predict(LeftEye)
        left_predictor = nmpy.argmax(predict_l_eye,axis=1)

        if(left_predictor[0]==1):
            label='Eyes_Opened'   
        if(left_predictor[0]==0):
            label='Eyes_Closed'
        break
    
    for (xaxs,yaxs,wdh,ht) in rightframes:
        RightEye=Img_frame[yaxs:yaxs+ht,xaxs:xaxs+wdh]
        cnt=cnt+1
        RightEye = cv2.cvtColor(RightEye,cv2.COLOR_BGR2GRAY)
        RightEye = cv2.resize(RightEye,(24,24))
        RightEye= RightEye/255
        RightEye=  RightEye.reshape(24,24,-1)
        RightEye = nmpy.expand_dims(RightEye,axis=0)
        predict_r_eye = Neural_net_model.predict(RightEye)
        right_predictor = nmpy.argmax(predict_r_eye,axis=1)

        if(right_predictor[0]==1):
            label='Eyes_Opened' 
        if(right_predictor[0]==0):
            label='Eyes_Closed'
        break

    
    for (xaxs,yaxs,wdh,ht) in fframes:
        cv2.rectangle(Img_frame, (xaxs,yaxs) , (xaxs+wdh,yaxs+ht) , (100,100,100) , 1 )

    if(right_predictor[0]==0 and left_predictor[0]==0):
        cv2.putText(Img_frame,"Eyes are Closed",(10,hgt-15), txtfont, 1,(0,0,0),1,cv2.LINE_AA)
        Blink_rate=Blink_rate+1
    else:
        cv2.putText(Img_frame,"Eyes are Opened",(10,hgt-15), txtfont, 1,(0,0,0),1,cv2.LINE_AA)
        Blink_rate=Blink_rate-1
    
        
    if(Blink_rate<0):
        Blink_rate=0   
    cv2.putText(Img_frame,'Blink Rate:'+str(Blink_rate),(400,hgt-15), txtfont, 1,(0,0,0),1,cv2.LINE_AA)
    if(Blink_rate>10):
      
        cv2.imwrite(os.path.join(Dirp,'image.jpg'),Img_frame)
        try:
            EyeAlert.play()
            
        except:  
            pass
        if(thickness_border<16):
            thickness_border= thickness_border+2
        else:
            thickness_border=thickness_border-2
            if(thickness_border<2):
                thickness_border=2
        cv2.rectangle(Img_frame,(0,0),(wd,hgt),(0,0,255),thickness_border) 
        cv2.putText(Img_frame,'ALERT!',(50,200), txtfont, 5,(0,0,255),thickness_border,cv2.LINE_AA)
    cv2.imshow('Img_frame',Img_frame)
  
    if cv2.waitKey(1)==27:
        break
        
capture.release()
cv2.destroyAllWindows() 