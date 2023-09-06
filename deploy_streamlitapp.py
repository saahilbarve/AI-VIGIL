import streamlit as st
from PIL import  Image
import tempfile
import cv2
from super_gradients.training import models
import torch
import numpy as np
import math
import streamlit as st
import time

def load_yolonas_process_each_image(image, confidence, st):

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = models.get('yolo_nas_s', pretrained_weights = "coco").to(device)


    classNames: ["moderate-accident", "object-accident", "severe-accident"]




    result = list(model.predict(image, conf=confidence))[0]
    bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
    confidences = result.prediction.confidence
    labels = result.prediction.labels.tolist()
    for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
        bbox = np.array(bbox_xyxy)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        x1, y1, x2, y2  = int(x1), int(y1), int(x2), int(y2)
        classname = int(cls)
        class_name = classNames[classname]
        conf = math.ceil((confidence*100))/100
        label = f'{class_name}{conf}'
        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255),3)
        cv2.rectangle(image, (x1, y1), c2, [255, 144, 30], -1, cv2.LINE_AA)
        cv2.putText(image, label, (x1, y1-2), 0, 1, [255, 255,255], thickness=1, lineType = cv2.LINE_AA)
    st.subheader('Output Image')
    st.image(image, channels = 'BGR', use_column_width=True)


def load_yolo_nas_process_each_frame(video_name, kpi1_text, kpi2_text, kpi3_text, stframe):
    cap = cv2.VideoCapture(video_name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_width = int(cap.get(3))

    frame_height = int(cap.get(4))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = models.get('yolo_nas_s', pretrained_weights="coco").to(device)

    count = 0
    prev_time = 0
    classNames: ["moderate-accident", "object-accident", "severe-accident"]


    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'O', 'S'), 10, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        count += 1
        if ret:
            result = list(model.predict(frame, conf=0.65))[0]
            bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
            confidences = result.prediction.confidence
            labels = result.prediction.labels.tolist()
            for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                classname = int(cls)
                class_name = classNames[classname]
                conf = math.ceil((confidence * 100)) / 100
                label = f'{class_name}{conf}'
                print("Frame N", count, "", x1, y1, x2, y2)
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.rectangle(frame, (x1, y1), c2, [255, 144, 30], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            stframe.image(frame, channels='BGR', use_column_width = True)
            current_time = time.time()

            fps = 1/(current_time - prev_time)

            prev_time = current_time

            kpi1_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(width)}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(height)}</h1>", unsafe_allow_html=True)

        else:
            break



def main():
    st.title("Object Detection with YOLO-NAS")
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left: -300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    app_mode = st.sidebar.selectbox('Choose the App Mode', ['About App', 'Run on Image', 'Run on Video'])

    if app_mode == 'About App':
        st.markdown('In this project i am using **YOLO-NAS** to do Object Detection on Images and Videos and we are using **StreamLit** to create a Graphical User Interface (GUI)')
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 300px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 300px;
                margin-left: -300px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.video('https://www.youtube.com/watch?v=rkTi6x5asLw&t=2149s&pp=ygUIeW9sbyBuYXM%3D')
        st.markdown('''
                    # About Me \n
                    Its Muhammad Moin, a computer vision enthuiast. Please check my YouTube Channel
                    - [YouTube] (https://www.youtube.com/@muhammadmoinfaisal/videos)
                    ''')

    elif app_mode == 'Run on Image':
        st.sidebar.markdown('---')
        confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0)
        st.sidebar.markdown('---')
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 300px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 300px;
                margin-left: -300px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

        #DEMO_IMAGE = 'Image\image3.png'

        if img_file_buffer is not None:
            img = cv2.imdecode(np.fromstring(img_file_buffer.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer))
        #else:
        #    img = cv2.imread(DEMO_IMAGE)
        #    image = np.array(Image.open(DEMO_IMAGE))
        #st.sidebar.text('Orignal Image')
        #st.sidebar.image(image)
            load_yolonas_process_each_image(img, confidence, st)
    elif app_mode == 'Run on Video':
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 300px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 300px;
                margin-left: -300px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.markdown('---')
        video_file_buffer = st.sidebar.file_uploader("Upload a Video", type = ["mp4", "avi", "mov", "asf"])

        #DEMO_VIDEO = 'Video/bikes.mp4'

        tffile = tempfile.NamedTemporaryFile(suffix= '.mp4', delete=False)

        if not video_file_buffer:
            tffile.name = 0
        else:
            tffile.write(video_file_buffer.read())
            demo_vid = open(tffile.name, 'rb')
            demo_bytes = demo_vid.read()
            st.sidebar.text('Input Video')
            st.sidebar.video(demo_bytes)
        stframe = st.empty()
        st.markdown("<hr/>", unsafe_allow_html = True)
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")
        with kpi2:
            st.markdown("**Width**")
            kpi2_text = st.markdown("0")
        with kpi3:
            st.markdown("**Height**")
            kpi3_text = st.markdown("0")
        st.markdown("<hr/>", unsafe_allow_html = True)
        load_yolo_nas_process_each_frame(tffile.name, kpi1_text,kpi2_text,  kpi3_text, stframe)

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass