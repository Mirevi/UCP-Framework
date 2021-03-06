from threading import Thread
from queue import Queue
import time
import numpy as np
import cv2

IMAGE_SIZE=512

# Cam IDs
camID_left_eye = "Testvideos/2020-08-20 23-12-02.mp4"
camID_right_eye = "Testvideos/2020-08-20 23-11-07.mp4"
camID_cnn = "CNN/2020-08-10 22-37-13.mp4"

# OSC Server (receive eye data "/faciallandmarks/eyes")
osc_server_ip = "127.0.0.1"
osc_server_port = 5005

# OSC Client (send landmarks)
osc_client_ip = "127.0.0.1"
osc_client_port = 9000

# CNN
cnn_model_path = "CNN/14_model__bs=8_lr=0.001_dr=0_op=adam_Weitwinkel3_OhneLampe3.pth"
cnn_params = dict(x=211,
                  y=440,
                  scale=148,
                  brightness=92)


def thread_osc(threadname, _queue_eyes):
    print("Start:" + threadname)

    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import BlockingOSCUDPServer

    def eyes_handler(address, *args):
        if not _queue_eyes.empty():
            _queue_eyes.get()
        _queue_eyes.put(args)

    dispatcher = Dispatcher()
    dispatcher.map("/faciallandmarks/eyes", eyes_handler)

    server = BlockingOSCUDPServer((osc_server_ip, osc_server_port), dispatcher)
    server.serve_forever()  # Blocks forever


def thread_eyebrow(threadname, _queue_eyebrown):
    print("Start:" + threadname)

    from Utils.EyebrowTracking import Tracker, TrackingPipeline

    video_left = cv2.VideoCapture(camID_left_eye)
    video_right = cv2.VideoCapture(camID_right_eye)

    cv2.namedWindow(threadname)

    tracker_left = [Tracker("Tracker 1", 95, 65, 10, 200),
                    Tracker("Tracker 2", 230, 40, 10, 135),
                    #Tracker("Tracker 3", 320, 50, 10, 200),
                    ]
    tracker_right = [Tracker("Tracker 4", 95, 65, 10, 200),
                     Tracker("Tracker 5", 230, 40, 10, 135),
                     #Tracker("Tracker 6", 320, 50, 10, 200),
                     ]
    tracking_pipeline_left = TrackingPipeline("Left", 110, 50, 520, 200)
    tracking_pipeline_right = TrackingPipeline("Right", 75, 0, 520, 200)

    timer = time.time()

    while video_left.isOpened() and video_right.isOpened():
        ret, frame_right = video_right.read()
        if not ret:
            video_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        ret, frame_left = video_left.read()
        if not ret:
            video_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_left = cv2.flip(frame_left, 1)

        tracked_poses_left, frame_left = tracking_pipeline_left(frame_left, tracker_left)
        tracked_poses_right, frame_right = tracking_pipeline_right(frame_right, tracker_right)

        tracked_poses = [0,  # tracked_poses_left[2],
                         tracked_poses_left[1],
                         tracked_poses_left[0],
                         tracked_poses_right[0],
                         tracked_poses_right[1],
                         0,  # tracked_poses_right[2]
                        ]

        if not _queue_eyebrown.empty():
            _queue_eyebrown.get()
        _queue_eyebrown.put(tracked_poses)

        # hist = cv2.calcHist([frame],[0],None,[256],[0,256])
        # cv2.imshow("hist", hist)

        frame_left = cv2.flip(frame_left, 1)

        cv2.imshow(threadname, np.concatenate((frame_left, frame_right), axis=1))
        if cv2.waitKey(1) == ord('q'):
            break

        #print(time.time() - timer)
        timer = time.time()

    video_left.release()
    video_right.release()
    cv2.destroyAllWindows()


def thread_hmc(threadname, _queue_hmc):
    print("Start:" + threadname)

    def nothing():
        pass

    import torch
    import Utils.Camera as cam
    from CNN.cnn import Net

    height = int(260 / 1.5 * 0.9)  # int(206 * 0.9)
    width = int(340 / 1.5 * 0.9)  # int(270 * 0.9)

    cv2.namedWindow('Window')
    cv2.createTrackbar('X', 'Window', cnn_params["x"], 450, nothing)
    cv2.createTrackbar('Y', 'Window', cnn_params["y"], 550, nothing)
    cv2.createTrackbar('Scale', 'Window', cnn_params["scale"], 200, nothing)
    cv2.createTrackbar('Brightness', 'Window', cnn_params["brightness"], 200, nothing)

    cam = cam.Camera("Weitwinkel")

    cap = cv2.VideoCapture(camID_cnn)  # (camID)  # 0 for webcam or path to video("Philipp_Grimaces.mp4") #

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(torch.zeros(1, 1, height, width), 0.1)
    model.load_state_dict(torch.load(cnn_model_path))
    model.to(device)
    model.eval()

    timer = 0

    while (True):
        _, frame = cap.read()
        frame = cam(frame)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Region of interest
        x = cv2.getTrackbarPos('X', 'Window')
        y = cv2.getTrackbarPos('Y', 'Window')
        scale = cv2.getTrackbarPos('Scale', 'Window')
        brightness = cv2.getTrackbarPos('Brightness', 'Window')

        x_min = int(x - width * scale / 200)
        x_max = int(x + width * scale / 200)
        y_min = int(y - height * scale / 200)
        y_max = int(y + height * scale / 200)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0))

        roi = frame[y_min:y_max, x_min:x_max]
        roi = cv2.resize(roi, (width, height))
        cv2.equalizeHist(roi, roi)
        roi = (roi * (brightness / 100))
        roi = np.clip(roi, 0, 255).astype("uint8")

        # cv2.flip(roi, 1, roi)

        # CNN
        input = torch.tensor(roi / 255, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        output = model(input.to(device))

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        x_prev = int(output[0][0] * width)
        y_prev = int(output[0][0 + 1] * height)
        for i in range(2, len(output[0]), 2):
            x = int(output[0][i] * width)
            y = int(output[0][i + 1] * height)
            # cv2.circle(frame, (x, y), 1, (255, 0, 0))
            if i != 22 and i != 32:
                cv2.line(roi, (x_prev, y_prev), (x, y), (0, 0, 255))
            x_prev = x
            y_prev = y

        roi = cv2.resize(roi, (x_max - x_min, y_max - y_min))
        frame[y_min:y_max, x_min:x_max] = roi

        cv2.imshow('Window', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not _queue_hmc.empty():
            _queue_hmc.get()
        _queue_hmc.put(output[0].cpu().detach().numpy().reshape(36, 2))

        #print(time.time() - timer)
        #timer = time.time()


def thread_fan(threadname, _queue_fan):
    print("Start:" + threadname)

    import torch
    import face_alignment
    import cv2
    from Utils.HeatmapDrawing import drawHeatmap
    from Utils.CropAndResize import cropAndResizeImageLandmarkBased

    imageSize = 256
    camID = 0
    cap = cv2.VideoCapture(0)#("1.Test - GesamtesGesichtMitWeitwinkelkameraVonUnten-02-08-20.mp4")  # 0 for webcam or path to video
    while not cap.isOpened():
        print("Error opening video stream or file")
        camID += 1
        cap = cv2.VideoCapture(camID)  # 0 for webcam or path to video
    print("Camera ID:", camID)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

    upperface = cv2.imread("winters-1919143_640.jpg")[0:250]

    while True:
        # Capture frame-by-frame
        has_frame, frame = cap.read()
        if not has_frame:
            print("continue: no frame")
            continue

        frame = np.concatenate((upperface, frame))

        cv2.imshow("Camera", frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #frame = cv2.resize(frame, (340, 260))
        #start = time.time()
        preds = fa.get_landmarks(frame)
        #print(time.time()-start)

        if preds == None:
            print("continue: no face")
            continue
            #preds = np.zeros((1, 68, 2))

        image, landmarks = cropAndResizeImageLandmarkBased(frame, imageSize, preds[0])

        if not _queue_fan.empty():
            _queue_fan.get()
        _queue_fan.put(landmarks)

        cv2.imshow("FAN", drawHeatmap(preds[0], imageSize))
        cv2.waitKey(1)


def thread_dummy(threadname, _queue_eyes):
    print("Start:" + threadname)
    while True:
        test = [1, 1, 1, 1, 1, 0]
        if not _queue_eyes.empty():
            _queue_eyes.get()
        _queue_eyes.put(test)
        time.sleep(0.001)


def thread_main(threadname, _queue_eyebrown, _queue_eyes, _queue_hmc):
    print("Start: " + threadname)

    from Utils.FacialLandmarkControl import FacialLandmarkController
    from pythonosc import udp_client

    flc = FacialLandmarkController(IMAGE_SIZE)
    osc_client = udp_client.SimpleUDPClient(osc_client_ip, osc_client_port)
    landmarks = np.zeros([70, 2])

    timer = 0

    #print("Waiting for data (" + threadname + ")")
    while True:

        data_eyebrown = list(_queue_eyebrown.get())
        data_eyes = _queue_eyes.get()
        #data_fan = _queue_fan.get()
        data_hmc = _queue_hmc.get()

        # print("Eyebrowns:", data_eyebrown)
        # print("Eye:", data_eyes)
        # print("Fan:", data_fan)
        # print("HMC:", data_hmc)
        timer = time.time()
        data_hmc = data_hmc * np.array([255, 127]) + np.array([0, 127])

        # FAN
        #landmarks[0:68, :] = data_fan

        # Eyebrown
        data_eyebrown = list(map(lambda y: y/5 + 15, data_eyebrown))

        landmarks[17, :] = np.array([45, data_eyebrown[2]])
        landmarks[18, :] = np.array([55, (2 * data_eyebrown[2] + data_eyebrown[1]) / 3])
        landmarks[19, :] = np.array([65, (data_eyebrown[2] + 2 * data_eyebrown[1]) / 3])
        landmarks[20, :] = np.array([75, (data_eyebrown[2] + data_eyebrown[1]) / 2])
        landmarks[21, :] = np.array([85, data_eyebrown[2]])

        landmarks[22, :] = np.array([170, data_eyebrown[3]])
        landmarks[23, :] = np.array([180, (data_eyebrown[3] + data_eyebrown[4])/2])
        landmarks[24, :] = np.array([190, (data_eyebrown[3] + 2 * data_eyebrown[4])/3])
        landmarks[25, :] = np.array([200, (2 * data_eyebrown[3] + data_eyebrown[4])/3])
        landmarks[26, :] = np.array([210, data_eyebrown[3]])

        # Eyes

        landmarks[36, :] = np.array([50, 60])
        landmarks[37, :] = np.array([60, 56]) + np.array([0, -3]) * data_eyes[2]
        landmarks[38, :] = np.array([80, 56]) + np.array([0, -3]) * data_eyes[2]
        landmarks[39, :] = np.array([90, 60])
        landmarks[40, :] = np.array([80, 64]) + np.array([0, 3]) * data_eyes[2]
        landmarks[41, :] = np.array([60, 64]) + np.array([0, 3]) * data_eyes[2]

        range_eye_X = 20
        range_eye_Y = 10
        landmarks[68, :] = np.array([70, 60]) + np.array([data_eyes[0] * range_eye_X - range_eye_X/2,
                                                          data_eyes[1] * range_eye_Y - range_eye_Y/2])

        landmarks[42, :] = np.array([165, 60])
        landmarks[43, :] = np.array([175, 56]) + np.array([0, -3]) * data_eyes[5]
        landmarks[44, :] = np.array([195, 56]) + np.array([0, -3]) * data_eyes[5]
        landmarks[45, :] = np.array([205, 60])
        landmarks[46, :] = np.array([195, 64]) + np.array([0, 3]) * data_eyes[5]
        landmarks[47, :] = np.array([175, 64]) + np.array([0, 3]) * data_eyes[5]

        landmarks[69, :] = np.array([185, 60]) + np.array([data_eyes[3] * range_eye_X - range_eye_X/2,
                                                           data_eyes[4] * range_eye_Y - range_eye_Y/2])

        # HMC
        landmarks[0, :] = data_hmc[0] + np.array([0, -30])
        landmarks[1, :] = data_hmc[0] + np.array([0, -20])
        landmarks[2, :] = data_hmc[0] + np.array([0, -10])
        landmarks[3:14, :] = data_hmc[0:11]
        landmarks[14, :] = data_hmc[10] + np.array([0, -10])
        landmarks[15, :] = data_hmc[10] + np.array([0, -20])
        landmarks[16, :] = data_hmc[10] + np.array([0, -30])
        landmarks[31:36, :] = data_hmc[11:16]
        landmarks[48:68, :] = data_hmc[16:36]

        # Nose
        landmarks[27, :] = np.array([127, 100 - 30])
        landmarks[28, :] = np.array([127, 100 - 20])
        landmarks[29, :] = np.array([127, 100 - 10])
        landmarks[30, :] = np.array([127, 100])

        if IMAGE_SIZE == 512:
            landmarks *= 2
        landmarks = flc(landmarks)
        print(time.time() - timer)
        osc_client.send_message("/landmarks", landmarks.reshape(-1).astype(int).tolist())



queue_eyebrown = Queue()
queue_eyes = Queue()
#queue_fan = Queue()
queue_hmc = Queue()

#thread1 = Thread(target=thread_osc, args=("Thread-OSC", queue_eyes))
#thread2 = Thread( target=thread_fan, args=("Thread-FAN", queue_fan))
thread2 = Thread(target=thread_hmc, args=("Thread-HMC", queue_hmc))
thread3 = Thread(target=thread_main, args=("Thread-Main", queue_eyebrown, queue_eyes, queue_hmc))
thread4 = Thread(target=thread_eyebrow, args=("Thread-Eyebrow", queue_eyebrown))
thread1 = Thread(target=thread_dummy, args=("Thread-DummyEye", queue_eyes))

thread1.start()
thread2.start()
thread3.start()
thread4.start()
#thread5.start()

thread1.join()
thread2.join()
thread3.join()
thread4.join()
