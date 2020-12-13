
![alt text](headerLogo.jpg "Header")

# Unmasking Communication Partner Framework

[[Paper]](https://arxiv.org/abs/2011.03630)
[[Video]](https://www.youtube.com/watch?v=Wa95qDPV8vk&feature=youtu.be)
[[Bibtex]](##Citation)

The UCP-Framework consists of the following parts:
- [RGBD-Face-Avatar-GAN](RGBD-Face-Avatar-GAN)
- [Facial-Landmark-OSC-Client](Facial-Landmark-OSC-Client)
- [Lower-Face-CNN](Lower-Face-CNN)

## Overview:

### [RGBD-Face-Avatar-GAN](RGBD-Face-Avatar-GAN)

The combination of Pix2Pix-GAN and RGBD-Images abels to generate personal face-avatars witch can be controlled 
by facial landmarks.

![alt text](RGBD-Face-Avatar-GAN/Images/Overview.png)

### [Facial-Landmark-OSC-Client](Facial-Landmark-OSC-Client)

The Facial-Landmark-OSC-Client detect facial landmarks from different input streams and merges the landmarks to a face 
representation. The whole face consists of 70 landmarks and is sent via OSC to an apllication with the ,,traced 
generator" from the RGBD-Face-Avatar-GAN.

### [Lower-Face-CNN](Lower-Face-CNN)

The Lower-Face-CNN enables the detection of facial landmarks **only** at the lower face.

![alt text](Lower-Face-CNN/Images/CNN.png)

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


