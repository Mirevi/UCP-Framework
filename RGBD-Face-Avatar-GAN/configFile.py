# Data
DatasetName = "1440pColorShot-Philipp"
FlipYAxis = False
IMAGE_SIZE = 256
FACETRACKING = "fan" # alternativ: "dlib"

# Pix2Pix
INPUT_CHANNEL = 1

# Depth Image
DEPTH_OFFSET = 300
DEPTH_MAX = 256 - 1

# IREyeTracking
IRET_Region = {
    "x": 500*2 + 60,
    "y": 320*2 -30,
    "width": 160*2,
    "height": 40*2
}
IRET_THRESHOLD = 88

# Test
TEST_VIDEO = "Data/Philipp-Known-Setting-RGBDFaceAvatarGAN-Datensatz03-09-2020/WIN_20201112_00_37_17_Pro.mp4" # OPTIONAL
TEST_INPUT = "Camera" # or "OSC"

USE_FLC = True
