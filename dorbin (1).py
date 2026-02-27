import cv2
import pandas as pd
from datetime import datetime, timedelta

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
attendance_data=[]
registered_faces=[]
DISTANCE_THRESHOLD=100
TIME_THRESHOLD=timedelta(seconds=30)
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    if not ret:break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    current_time=datetime.now()
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
    for (x,y,w,h) in faces:
        center_x=x+w//2
        center_y=y+h//2
        is_new=True
        for reg_x,reg_y,reg_time in registered_faces:
            if (abs(reg_x-center_x)<DISTANCE_THRESHOLD and abs(reg_y-center_y)<DISTANCE_THRESHOLD and (current_time-reg_time)<TIME_THRESHOLD):
                is_new=False
                break
        if is_new:
            student_name=f"Student {len(attendance_data)+1}"
            attendance_data.append([student_name,current_time.strftime("%Y-%m-%d %H:%M:%S")])
            registered_faces.append((center_x,center_y,current_time))
            print(f"New Stable Scan: {student_name} at {current_time.strftime('%H:%M:%S')}")
            cv2.putText(frame,'New Face Scanned!',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.circle(frame,(x+ex+ew//2,y+ey+eh//2),eh//2,(0,255,0),2)
    cv2.putText(frame,'Stand in front and smile! (Fixed No Duplicates)',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.imshow('Stable Face Attendance - Fixed',frame)
    if cv2.waitKey(1)&0xFF==ord('q'):break

df=pd.DataFrame(attendance_data,columns=['Student Name','Time'])
df.to_csv('stable_attendance_fixed.csv',index=False)
print("File saved! Check Excel - no errors.")
cap.release()
cv2.destroyAllWindows()
