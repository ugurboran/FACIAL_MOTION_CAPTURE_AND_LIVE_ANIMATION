import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0) # create an object named videocapture and index means which camera will it be used.

detector = dlib.get_frontal_face_detector() #dlib’s pre-trained face detector

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #loads the facial landmark predictor

while True:

    _, frame = cap.read() #Capture frame by frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #image has 3 different layer so it changes into one layer which is gray

    faces = detector(gray)
    for face in faces:

        #this part is for face detection
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)


        #This part is for facial landmarks detection, every landmark has x and y coordinates so this part should map with the avatar mesh in Blender.
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)


    cv2.imshow("Frame", frame) # display the frame

    key = cv2.waitKey(1) # 27 means 'ESC', When 'ESC' key is push, then break the loop.
    if key == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()