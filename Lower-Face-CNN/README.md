# Lower-Face-CNN

## Install
Install pytorch from pytorch.org and run pip with the requiements.txt

## Usage
1.) Add a path to a video to PATH2VID in DataTool.py (usually data is in the dir /data/). The video must be made by a wide angle camera that shows the face in a very short distance close up.
2.) Run DataTool.py. It loads the video, detect facial landmarks and automatically adjust a bounding box of the lower face. DataTool.py produces da dataset of landmarks based on the bounding box area.
3.) Add the path to the landmark_cropped.csv from PATH2VID to Datset.py in the FaceLandmarkDataset(PUT PATH HERE)
4.) Run train.py



