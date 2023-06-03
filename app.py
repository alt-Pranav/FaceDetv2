import streamlit as st
import cv2
from mtcnn import MTCNN

detector = MTCNN()

def detect_bounding_box(frame):
    # detect faces in the image
    faces = detector.detect_faces(frame)
    #for face in faces:
    #   print(face) 
    
    for face in faces:
        x, y, width, height = face['box']
        conf = face['confidence']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 4)
        cv2.putText(frame, str(conf)[:4], (x+width, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return faces

def main():

    st.title("Face Detector")

    checked = st.checkbox(label="Start Video Feed", value=False)

    if checked:

        FRAME_WINDOW = st.image([])
        videoFeed = cv2.VideoCapture(-1)

        while True:
            _, frame = videoFeed.read()

            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                faces = detect_bounding_box(frame=frame)

                FRAME_WINDOW.image(frame)
            
            else:
                st.write("No image detected, please check webcam")

    else:
        st.write("Click to start")


if __name__ == "__main__":
    main()