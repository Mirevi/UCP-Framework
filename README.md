
![alt text](headerLogo.jpg "Header")

# Unmasking Communication Partner Framework

[[Paper]](https://arxiv.org/abs/2011.03630)
[[Video]](https://www.youtube.com/watch?v=Wa95qDPV8vk&feature=youtu.be)
[[Bibtex]](##Citation)

The UCP-Framework consists of the following parts:
- [RGBD-Face-Avatar-GAN](RGBD-Face-Avatar-GAN)
- [Facial-Landmark-OSC-Client](Facial-Landmark-OSC-Client)
- [Lower-Face-CNN](Lower-Face-CNN)
- [MireviMotionHub](https://github.com/Mirevi/MotionHub)
- [ClayMore-Engine](https://github.com/Mirevi/ClayMore)

## Overview:

### [RGBD-Face-Avatar-GAN](RGBD-Face-Avatar-GAN)

This GAN is based on the Pix2Pix-GAN and extended with a additional channel for processing RGBD images. It allows the generation of three-dimensional face avatars which can be controlled by facial landmarks.

![alt text](RGBD-Face-Avatar-GAN/Images/Overview.png)

### [Facial-Landmark-OSC-Client](Facial-Landmark-OSC-Client)

The Facial-Landmark-OSC-Client detect facial landmarks from different input streams and merges the landmarks to a face 
representation. The whole face consists of 70 landmarks and is sent via OSC to an apllication with the ,,traced 
generator" from the RGBD-Face-Avatar-GAN.

### [Lower-Face-CNN](Lower-Face-CNN)

The Lower-Face-CNN enables the detection of facial landmarks **only** at the lower face.

![alt text](Lower-Face-CNN/Images/CNN.png)

### [MireviMotionHub](https://github.com/Mirevi/MotionHub)
The MotionHub supports in the branch "GANExport" a window (Menu->GAN Capture) for recording the face with a Microsoft Azure Kinect. This way, a data set can be acquired and later be processed by the RGBD-Face-Avatar-GAN. In future releases of the MotionHub, a convienient installer will be provided with the GANCapture feature. 

### [ClayMore Engine](https://github.com/Mirevi/ClayMore)
This is the engine which executes the trained generator module and enables the basics for a potential real time telepresence system with three dimensional faces. Please send a request to Philipp Ladwig for access to the source code (see paper for address). We are currently maintaining the source and we are planning to release a open source version soon.


## Citation
```
@misc{ladwig2020unmasking,
      title={Unmasking Communication Partners: A Low-Cost AI Solution for Digitally Removing Head-Mounted Displays in VR-Based Telepresence}, 
      author={Philipp Ladwig and Alexander Pech and Ralf Dörner and Christian Geiger},
      year={2020},
      eprint={2011.03630},
      archivePrefix={arXiv},
      primaryClass={cs.GR}
}
```


