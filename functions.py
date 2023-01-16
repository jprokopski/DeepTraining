import cv2
import mediapipe as mp
import numpy as np
import simplejpeg

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def biceps(cap):
    counter = 0 
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            NULL, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            try:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                left_elbow_text = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x-0.01,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y+0.01]

                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                
                if left_angle > 140 and right_angle > 140:
                        stage = "down"

                if left_angle < 30 and stage =='down' and right_angle < 30:
                        stage="up"
                        counter +=1

            except:
                pass

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            cv2.line(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), tuple(np.multiply(right_wrist, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.circle(image, tuple(np.multiply(right_wrist, [1280, 720]).astype(int)), 20, (0,128,0), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_wrist, [1280, 720]).astype(int)), 25, (0,128,0), 2)
            cv2.circle(image, tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), 30, (0,128,0), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), 35, (0,128,0), 2)
            cv2.circle(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), 20, (0,128,0), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), 25, (0,128,0), 2)


            cv2.line(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), tuple(np.multiply(left_wrist, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.circle(image, tuple(np.multiply(left_wrist, [1280, 720]).astype(int)), 20, (0,128,0), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_wrist, [1280, 720]).astype(int)), 25, (0,128,0), 2)
            cv2.circle(image, tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), 30, (0,128,0), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), 35, (0,128,0), 2)
            cv2.circle(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), 20, (0,128,0), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), 25, (0,128,0), 2)


            left_per = np.interp(left_angle, (30, 160), (100, 0))
            left_bar = np.interp(left_angle, (30, 160), (100, 650))
            left_color = (0, 165, 255)
            if left_per == 100:
                left_color = (0, 255, 0)
  
            if left_per == 0:
                left_color = (0, 0, 255)

            right_per = np.interp(right_angle, (30, 160), (100, 0))
            right_bar = np.interp(right_angle, (30, 160), (100, 650))
            right_color = (0, 165, 255)
            if right_per == 100:
                right_color = (0, 255, 0)
  
            if right_per == 0:
                right_color = (0, 0, 255)

            whole_color = (0, 165, 255)
            if left_per == 100 and right_per == 100:
                whole_color = (0, 255, 0)
  
            if left_per == 0 and right_per == 0:
                whole_color = (0, 0, 255)

            cv2.rectangle(image, (100, 100), (175, 650), right_color, 3)
            cv2.rectangle(image, (100, int(right_bar)), (175, 650), right_color, cv2.FILLED)

            cv2.rectangle(image, (475,0), (800,100), whole_color, -1)
            cv2.putText(image, 'REPS', (510,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (510,85) if counter < 10 else (490,85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (675,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (620,85) if stage == "down" else (660,85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (1100, 100), (1175, 650), left_color, 3)
            cv2.rectangle(image, (1100, int(left_bar)), (1175, 650), left_color, cv2.FILLED)  

            cv2.putText(image, str(round(left_angle)), 
                        tuple(np.multiply(left_elbow_text, [1280, 720]).astype(int)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            cv2.putText(image, str(round(right_angle)), 
                        tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                

            jpeg = simplejpeg.encode_jpeg(image, colorspace = "BGR")
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')
            
def przysiad(cap):
    counter = 0 
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            NULL, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            try:
                landmarks = results.pose_landmarks.landmark
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                left_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_angle = calculate_angle(right_hip, right_knee, right_ankle)
                

                if left_angle > 160 and right_angle > 160 and stage=="down":
                        stage = "up"
                        counter += 1
                elif left_angle > 160 and right_angle > 160:
                        stage = "up"

                if left_angle < 120 and stage =='up' and right_angle < 120:
                        stage = "down"

            except:
                pass

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.circle(image, tuple(np.multiply(right_hip, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_hip, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_knee, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_knee, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_ankle, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_ankle, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.line(image, tuple(np.multiply(right_hip, [1280, 720]).astype(int)), tuple(np.multiply(right_knee, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(right_knee, [1280, 720]).astype(int)), tuple(np.multiply(right_ankle, [1280, 720]).astype(int)), (255, 255, 255), 3)


            cv2.circle(image, tuple(np.multiply(left_hip, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_hip, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_knee, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_knee, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_ankle, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_ankle, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.line(image, tuple(np.multiply(left_hip, [1280, 720]).astype(int)), tuple(np.multiply(left_knee, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(left_knee, [1280, 720]).astype(int)), tuple(np.multiply(left_ankle, [1280, 720]).astype(int)), (255, 255, 255), 3)

            left_per = np.interp(left_angle, (120, 160), (0, 100))
            left_bar = np.interp(left_angle, (120, 160), (100, 650))
            left_color = (0, 165, 255)
            if left_per == 100:
                left_color = (0, 0, 255)
  
            if left_per == 0:
                left_color = (0, 255, 0)

            right_per = np.interp(right_angle, (120, 160), (0, 100))
            right_bar = np.interp(right_angle, (120, 160), (100, 650))
            right_color = (0, 165, 255)
            if right_per == 100:
                right_color = (0, 0, 255)
  
            if right_per == 0:
                right_color = (0, 255, 0)

            whole_color = (0, 165, 255)
            if left_per == 100 and right_per == 100:
                whole_color = (0, 255, 0)
  
            if left_per == 0 and right_per == 0:
                whole_color = (0, 0, 255)

            cv2.rectangle(image, (100, 100), (175, 650), right_color, 3)
            cv2.rectangle(image, (100, int(right_bar)), (175, 650), right_color, cv2.FILLED)

            cv2.rectangle(image, (475,0), (800,100), whole_color, -1)
            cv2.putText(image, 'REPS', (510,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (510,85) if counter < 10 else (490,85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (675,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (620,85) if stage == "down" else (660,85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (1100, 100), (1175, 650), left_color, 3)
            cv2.rectangle(image, (1100, int(left_bar)), (1175, 650), left_color, cv2.FILLED) 
              
            cv2.putText(image, str(round(left_angle)), 
                        tuple(np.multiply(left_knee, [1330, 730]).astype(int)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            cv2.putText(image, str(round(right_angle)), 
                        tuple(np.multiply(right_knee, [1120, 720]).astype(int)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
        
            jpeg = simplejpeg.encode_jpeg(image, colorspace = "BGR")
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')

def pompka(cap):
    counter = 0 
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            NULL, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            try:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]


                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                big_angle = calculate_angle(left_wrist, left_shoulder, left_ankle)
                

                if left_angle > 150 and right_angle > 150 and big_angle > 30 and stage == "down":
                        stage = "up"
                        counter += 1
                elif left_angle > 150 and right_angle > 150 and big_angle > 30:
                        stage = "up"

                if left_angle < 110 and stage =='up' and right_angle < 110 and big_angle > 30:
                        stage = "down"

            except:
                pass

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.circle(image, tuple(np.multiply(right_wrist, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_wrist, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_knee, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_knee, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_hip, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_hip, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_wrist, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_wrist, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_ankle, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_ankle, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.line(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), tuple(np.multiply(right_wrist, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(right_hip, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(right_knee, [1280, 720]).astype(int)), tuple(np.multiply(right_hip, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(right_knee, [1280, 720]).astype(int)), tuple(np.multiply(right_ankle, [1280, 720]).astype(int)), (255, 255, 255), 3)

            cv2.circle(image, tuple(np.multiply(left_wrist, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_wrist, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_knee, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_knee, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_hip, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_hip, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_ankle, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_ankle, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.line(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), tuple(np.multiply(left_wrist, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(left_hip, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(left_knee, [1280, 720]).astype(int)), tuple(np.multiply(left_hip, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(left_knee, [1280, 720]).astype(int)), tuple(np.multiply(left_ankle, [1280, 720]).astype(int)), (255, 255, 255), 3)

            left_per = np.interp(left_angle, (110, 150), (0, 100))
            left_bar = np.interp(left_angle, (110, 150), (100, 650))
            left_color = (0, 165, 255)
            if left_per == 100:
                left_color = (0, 0, 255)
  
            if left_per == 0:
                left_color = (0, 255, 0)

            right_per = np.interp(right_angle, (110, 150), (0, 100))
            right_bar = np.interp(right_angle, (110, 150), (100, 650))
            right_color = (0, 165, 255)
            if right_per == 100:
                right_color = (0, 0, 255)
  
            if right_per == 0:
                right_color = (0, 255, 0)

            whole_color = (0, 165, 255)
            if left_per == 100 and right_per == 100:
                whole_color = (0, 255, 0)
  
            if left_per == 0 and right_per == 0:
                whole_color = (0, 0, 255)

            cv2.rectangle(image, (100, 100), (175, 650), right_color, 3)
            cv2.rectangle(image, (100, int(right_bar)), (175, 650), right_color, cv2.FILLED)

            cv2.rectangle(image, (475,0), (800,100), whole_color, -1)
            cv2.putText(image, 'REPS', (510,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (510,85) if counter < 10 else (490,85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (675,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (620,85) if stage == "down" else (660,85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (1100, 100), (1175, 650), left_color, 3)
            cv2.rectangle(image, (1100, int(left_bar)), (1175, 650), left_color, cv2.FILLED) 
              
            cv2.putText(image, str(round(left_angle)), 
                        tuple(np.multiply(left_elbow, [1330, 730]).astype(int)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            cv2.putText(image, str(round(right_angle)), 
                        tuple(np.multiply(right_elbow, [1120, 720]).astype(int)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            jpeg = simplejpeg.encode_jpeg(image, colorspace = "BGR")
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')

def brzuszki(cap):
    counter = 0 
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            NULL, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            try:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                left_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                right_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                

                if left_angle > 120 and right_angle > 120 and stage == "up":
                        stage = "down"
                        counter +=1
                elif left_angle > 120 and right_angle > 120:
                        stage = "down"

                if left_angle < 70 and stage =='down' and right_angle < 70:
                        stage = "up"


            except:
                pass

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.circle(image, tuple(np.multiply(right_knee, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_knee, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_hip, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_hip, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.line(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(right_hip, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(right_knee, [1280, 720]).astype(int)), tuple(np.multiply(right_hip, [1280, 720]).astype(int)), (255, 255, 255), 3)

            cv2.circle(image, tuple(np.multiply(left_knee, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_knee, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_hip, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_hip, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.line(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(left_hip, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(left_knee, [1280, 720]).astype(int)), tuple(np.multiply(left_hip, [1280, 720]).astype(int)), (255, 255, 255), 3)

            left_per = np.interp(left_angle, (70, 120), (100, 0))
            left_bar = np.interp(left_angle, (70, 120), (100, 650))
            left_color = (0, 165, 255)
            if left_per == 100:
                left_color = (0, 255, 0)
  
            if left_per == 0:
                left_color = (0, 0, 255)

            right_per = np.interp(right_angle, (70, 120), (100, 0))
            right_bar = np.interp(right_angle, (70, 120), (100, 650))
            right_color = (0, 165, 255)
            if right_per == 100:
                right_color = (0, 255, 0)
  
            if right_per == 0:
                right_color = (0, 0, 255)

            whole_color = (0, 165, 255)
            if left_per == 100 and right_per == 100:
                whole_color = (0, 255, 0)
  
            if left_per == 0 and right_per == 0:
                whole_color = (0, 0, 255)

            cv2.rectangle(image, (100, 100), (175, 650), right_color, 3)
            cv2.rectangle(image, (100, int(right_bar)), (175, 650), right_color, cv2.FILLED)

            cv2.rectangle(image, (475,0), (800,100), whole_color, -1)
            cv2.putText(image, 'REPS', (510,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (510,85) if counter < 10 else (490,85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (675,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (620,85) if stage == "down" else (660,85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (1100, 100), (1175, 650), left_color, 3)
            cv2.rectangle(image, (1100, int(left_bar)), (1175, 650), left_color, cv2.FILLED) 

            cv2.putText(image, str(round(left_angle)), 
                        tuple(np.multiply(left_hip, [1330, 730]).astype(int)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            cv2.putText(image, str(round(right_angle)), 
                        tuple(np.multiply(right_hip, [1120, 720]).astype(int)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            jpeg = simplejpeg.encode_jpeg(image, colorspace = "BGR")
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')

def military(cap):
    counter = 0 
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            NULL, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            try:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                left_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                right_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                

                if left_angle > 170 and right_angle > 170:
                        stage = "up"

                if left_angle < 100 and stage =='up' and right_angle < 100:
                        stage="down"
                        counter +=1

            except:
                pass

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.circle(image, tuple(np.multiply(right_wrist, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_wrist, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_hip, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_hip, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.line(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(right_hip, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), tuple(np.multiply(right_wrist, [1280, 720]).astype(int)), (255, 255, 255), 3)


            cv2.circle(image, tuple(np.multiply(left_wrist, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_wrist, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_hip, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_hip, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.line(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(left_hip, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), tuple(np.multiply(left_wrist, [1280, 720]).astype(int)), (255, 255, 255), 3)

            left_per = np.interp(left_angle, (100, 170), (0, 100))
            left_bar = np.interp(left_angle, (100, 170), (650, 100))
            left_color = (0, 165, 255)
            if left_per == 100:
                left_color = (0, 255, 0)
  
            if left_per == 0:
                left_color = (0, 0, 255)

            right_per = np.interp(right_angle, (100, 170), (0, 100))
            right_bar = np.interp(right_angle, (100, 170), (650, 100))
            right_color = (0, 165, 255)
            if right_per == 100:
                right_color = (0, 255, 0)
  
            if right_per == 0:
                right_color = (0, 0, 255)

            whole_color = (0, 165, 255)
            if left_per == 100 and right_per == 100:
                whole_color = (0, 255, 0)
  
            if left_per == 0 and right_per == 0:
                whole_color = (0, 0, 255)

            cv2.rectangle(image, (100, 100), (175, 650), right_color, 3)
            cv2.rectangle(image, (100, int(right_bar)), (175, 650), right_color, cv2.FILLED)

            cv2.rectangle(image, (475,0), (800,100), whole_color, -1)
            cv2.putText(image, 'REPS', (510,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (510,85) if counter < 10 else (490,85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (675,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (620,85) if stage == "down" else (660,85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (1100, 100), (1175, 650), left_color, 3)
            cv2.rectangle(image, (1100, int(left_bar)), (1175, 650), left_color, cv2.FILLED) 
              
            cv2.putText(image, str(round(left_angle)), 
                        tuple(np.multiply(left_shoulder, [1330, 730]).astype(int)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            cv2.putText(image, str(round(right_angle)), 
                        tuple(np.multiply(right_shoulder, [1120, 720]).astype(int)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            jpeg = simplejpeg.encode_jpeg(image, colorspace = "BGR")
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')

def wznosy(cap):
    counter = 0 
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            NULL, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            try:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                left_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                right_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                

                if left_angle > 80 and right_angle > 80:
                        stage = "up"

                if left_angle < 20 and stage =='up' and right_angle < 20:
                        stage="down"
                        counter +=1

            except:
                pass

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.circle(image, tuple(np.multiply(right_wrist, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_wrist, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(right_hip, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(right_hip, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.line(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(right_hip, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), tuple(np.multiply(right_wrist, [1280, 720]).astype(int)), (255, 255, 255), 3)


            cv2.circle(image, tuple(np.multiply(left_wrist, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_wrist, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.circle(image, tuple(np.multiply(left_hip, [1280, 720]).astype(int)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(left_hip, [1280, 720]).astype(int)), 15, (0, 0, 255), 2)
            cv2.line(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(left_hip, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)), tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), (255, 255, 255), 3)
            cv2.line(image, tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), tuple(np.multiply(left_wrist, [1280, 720]).astype(int)), (255, 255, 255), 3)

            left_per = np.interp(left_angle, (20, 80), (0, 100))
            left_bar = np.interp(left_angle, (20, 80), (650, 100))
            left_color = (0, 165, 255)
            if left_per == 100:
                left_color = (0, 255, 0)
  
            if left_per == 0:
                left_color = (0, 0, 255)

            right_per = np.interp(right_angle, (20, 80), (0, 100))
            right_bar = np.interp(right_angle, (20, 80), (650, 100))
            right_color = (0, 165, 255)
            if right_per == 100:
                right_color = (0, 255, 0)
  
            if right_per == 0:
                right_color = (0, 0, 255)

            whole_color = (0, 165, 255)
            if left_per == 100 and right_per == 100:
                whole_color = (0, 255, 0)
  
            if left_per == 0 and right_per == 0:
                whole_color = (0, 0, 255)

            cv2.rectangle(image, (100, 100), (175, 650), right_color, 3)
            cv2.rectangle(image, (100, int(right_bar)), (175, 650), right_color, cv2.FILLED)

            cv2.rectangle(image, (475,0), (800,100), whole_color, -1)
            cv2.putText(image, 'REPS', (510,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (510,85) if counter < 10 else (490,85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (675,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (620,85) if stage == "down" else (660,85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (1100, 100), (1175, 650), left_color, 3)
            cv2.rectangle(image, (1100, int(left_bar)), (1175, 650), left_color, cv2.FILLED) 
              
            cv2.putText(image, str(round(left_angle)), 
                        tuple(np.multiply(left_shoulder, [1330, 730]).astype(int)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            cv2.putText(image, str(round(right_angle)), 
                        tuple(np.multiply(right_shoulder, [1120, 720]).astype(int)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            jpeg = simplejpeg.encode_jpeg(image, colorspace = "BGR")
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')