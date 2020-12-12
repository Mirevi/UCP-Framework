import numpy as np
import cv2
import cv2 as cv
import math
from pythonosc import udp_client

def nothing(x):
    pass

"""
def image_processing(frame):
    a = cv2.getTrackbarPos('A', 'frame', nothing)
    b = cv2.getTrackbarPos('B', 'frame', nothing)
    c = cv2.getTrackbarPos('C', 'frame', nothing)

    blocksize = lambda i: 3 + i * 2

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # frame = 255 - frame

    # Helligkeits ausgleich
    grad = np.repeat(np.tile(np.linspace(0, 1, frame.shape[1]), (frame.shape[0], 1))[:, :, np.newaxis], 3, axis=2)
    grad = (grad * 255).astype("uint8")
    grad = cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
    frame = (frame + 42 / 255 * grad).astype("uint8")

    #frame = (np.sin(frame * math.pi / 255) * 127.5 + 127.5).astype("uint8")

    # Maske
    #mask = np.zeros((480, 720))
    #mask = cv2.ellipse(mask, (300, 220), (250, 200), 15, 0, 360, 255, -1)
    #frame = frame * (mask / 255).astype("uint8")

    frame = np.clip(frame, 140, 190)
    frame = cv2.equalizeHist(frame)

    frame = cv2.GaussianBlur(frame, (blocksize(a), blocksize(a)), 0)

    #kernel = np.ones((b, b), np.uint8)
    #frame = cv.erode(frame, kernel, iterations=1)


    #frame = cv2.GaussianBlur(frame, (blocksize(a), blocksize(a)), 0)
    #frame = cv2.GaussianBlur(frame, (blocksize(a), blocksize(a)), 0)

    #frame = frame/255
    #for i in range(0, 2):
    #    frame *= frame
    #frame = (frame * 255).astype("uint8")

    if c > 0:
        ret, frame = cv2.threshold(frame, c, 255, cv.THRESH_BINARY)


    # frame = cv.dilate(frame, kernel, iterations=1)
    # frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    # frame = cv2.fastNlMeansDenoising(frame)
    # frame = cv2.bilateralFilter(frame, 9, 75, 75)
    # frame = cv2.Laplacian(frame,cv2.CV_64F)
    # frame = cv2.medianBlur(frame, blocksize(b))
    # frame = cv2.GaussianBlur(frame, (blocksize(b), blocksize(b)), 0)
    # frame = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blocksize(b), blocksize(c))
    # blur = cv2.GaussianBlur(frame, (blocksize(a), blocksize(a)), 0)
    # ret3, frame = cv2.threshold(frame, b, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # frame = cv2.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, b)
    # frame = cv2.Canny(frame, a, b)
    # kernel = np.ones((5, 5), np.uint8)
    # frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)

    #frame = cv2.Canny(frame,c,200)



    return frame
"""

class Tracker:

    def __init__(self, name: str, x: int, y: int, w: int, h: int):
        self.name = name
        self.x = x
        self.y = y
        self.h = h
        self.w = w

    def __call__(self, frame_eyebrown, frame = None, offset_x=0, offset_y=0):
        cv2.rectangle(frame_eyebrown, (self.x, self.y), (self.x + self.w, self.y + self.h), (127, 127, 127))
        frame_eyebrown_tracking = frame_eyebrown[self.y:self.y + self.h, self.x:self.x + self.w]
        #cv2.imshow(self.name, frame_eyebrown_tracking)

        white = np.where(frame_eyebrown_tracking == 255)
        if white[0] != []:
            tracked_y = int(np.median(white[0]))
        else:
            tracked_y = 135

        tracked_y += self.y
        #tracked_y = int(np.argmax(frame_eyebrown_tracking) / self.w) + self.y
        frame_eyebrown[tracked_y:tracked_y + self.w, self.x:self.x + self.w] = 127
        frame[tracked_y+offset_y:tracked_y+offset_y + self.w, self.x+offset_x:self.x+offset_x + self.w] = (0, 0, 255)

        return tracked_y, frame_eyebrown, frame

class TrackingPipeline:
    def __init__(self, name, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.window_name = name
        cv2.namedWindow(self.window_name)

        cv2.createTrackbar('Blur', self.window_name, 6, 255, nothing) #33
        self.blocksize = lambda i: 3 + i * 2
        cv2.createTrackbar('Brightness', self.window_name, 0, 255, nothing) #255 # 22
        cv2.createTrackbar('Threshold', self.window_name, 140, 255, nothing)  #170# 165

    def __call__(self, frame, tracker):
        # ROI - Eyebrown
        frame_eyebrown_bgr = frame[self.y:self.y+self.h, self.x:self.x+self.w, :]#frame[0:200, 75:520, :]
        frame_eyebrown = cv2.cvtColor(frame_eyebrown_bgr, cv2.COLOR_BGR2GRAY)
        frame_eyebrown = cv2.equalizeHist(frame_eyebrown)

        kernel = cv2.getTrackbarPos('Blur', self.window_name, )
        frame_eyebrown = cv2.GaussianBlur(frame_eyebrown, (self.blocksize(kernel), self.blocksize(kernel)), 0)

        brightness = cv2.getTrackbarPos('Brightness', self.window_name)
        # _, w = frame_eyebrown.shape
        # frame_eyebrown[:, int(w/2):w] += brightness
        grad = np.repeat(
            np.tile(np.linspace(0, 1, frame_eyebrown.shape[1]), (frame_eyebrown.shape[0], 1))[:, :, np.newaxis], 3,
            axis=2)
        grad = (grad * 255).astype("uint8")
        grad = cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
        frame_eyebrown = (frame_eyebrown + brightness / 255 * grad)
        frame_eyebrown = np.clip(frame_eyebrown, 0, 255).astype("uint8")

        threshold = cv2.getTrackbarPos('Threshold', self.window_name, )
        if threshold != 0:
            ret, frame_eyebrown = cv2.threshold(frame_eyebrown, threshold, 255, cv.THRESH_BINARY_INV)

        tracked_poses = []
        for t in tracker:
            trackpos, frame_eyebrown, frame = t(frame_eyebrown, frame, self.x, self.y)
            tracked_poses.append(trackpos)

        #stereo_frame_marked.append(frame)
        cv2.imshow(self.window_name, frame_eyebrown)
        return tracked_poses, frame

if __name__ == "__main__":

    osc_client = udp_client.SimpleUDPClient("127.0.0.1", 5005)

    video_right = cv2.VideoCapture("../Testvideos/3. EyeBrowTest7August.mp4")
    video_left = cv2.VideoCapture("../Testvideos/4. EyeBrowTest17August.mp4")
    cv2.namedWindow('frame')

    # cv2.createTrackbar('A', 'frame', 21, 255, nothing)
    # cv2.createTrackbar('B', 'frame', 0, 255, nothing)
    # cv2.createTrackbar('C', 'frame', 100, 255, nothing)
    # cv2.createTrackbar('D', 'frame', 1, 2, nothing)

    tracker_left = [Tracker("Tracker 1", 95, 65, 10, 200),
                    Tracker("Tracker 2", 230, 50, 10, 200),
                    Tracker("Tracker 3", 320, 50, 10, 200),
                   ]
    tracker_right = [Tracker("Tracker 4", 95, 65, 10, 200),
                     Tracker("Tracker 5", 230, 50, 10, 100),
                     Tracker("Tracker 6", 320, 50, 10, 200),
                    ]
    tracking_pipeline_left = TrackingPipeline("Left", 110, 40, 520, 200)
    tracking_pipeline_right = TrackingPipeline("Right", 75, 0, 520, 200)

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

        tracked_poses = [tracked_poses_left[2],
                         tracked_poses_left[1],
                         tracked_poses_left[0],
                         tracked_poses_right[0],
                         tracked_poses_right[1],
                         tracked_poses_right[2]
                         ]

        osc_client.send_message("/faciallandmarks/eyebrows", tracked_poses)
        tracked_poses = []

        #hist = cv2.calcHist([frame],[0],None,[256],[0,256])
        #cv2.imshow("hist", hist)

        frame_left = cv2.flip(frame_left, 1)

        cv2.imshow('frame', np.concatenate((frame_left, frame_right), axis=1))
        stereo_frame_marked = []
        if cv2.waitKey(1) == ord('q'):
            break
    video_left.release()
    video_right.release()
    cv2.destroyAllWindows()
