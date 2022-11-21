# Are your moves any good?

**Hi there! 'Dance Sync', originally called 'Pose' is a program that allows you to compare your dancing to your model dancer's.** <br />

When learning a dance, it can be hard to pinpoint where you're going wrong by just looking in the mirror. Instead, record a video of your dancing and plug it into 'Dance Sync' - it will analyse your movement and compare it to the video of the orginial dancer. <br />

You will get a comparison video, with alerts where you are not in sync and a final score at the end. <br />

![](https://github.com/Mruchus/dance-sync-analysis/blob/main/comparisonsample.gif)

[<img src="https://github.com/Mruchus/dance-sync-analysis/blob/main/finalscoresample.png" width="600"/>](https://github.com/Mruchus/dance-sync-analysis/blob/main/finalscoresample.png) <br />

Happy dancing! <br />

## How does it work?

### 1. How do we compare the dancers?

### 2. How do we know when to start comparing the two videos?

 🚧 *Under construction* 🚧 <br />

## Want to try it out?

I am working on making this accessible for everyone: **non-programmers included!** Unfortunately, I'm not quite there yet so you will need to install a bunch of packages (sorry!) and run the code on **macOS 12.5.1**. <br />

### Dependencies

Have these installed these before running:
* OpenCV2 https://pypi.org/project/opencv-python/
* MediaPipe https://google.github.io/mediapipe/getting_started/install.html#installing-on-macos
* Numpy https://numpy.org/install/
* Praat https://www.fon.hum.uva.nl/praat/download_mac.html
* Ffmpeg https://ffmpeg.org/download.html <br />

### Executing
Download the folder and run the main python file called **'sync_final.py'**. Make sure the videos are in **.mov** format and are the same dimension i.e. 1280x720 px

## Acknowledgments
* Inspiration: https://www.youtube.com/watch?v=zxhXj72WClE
* Useful documentation of how pose estimation works: https://google.github.io/mediapipe/solutions/pose.html
* (Dr Spyros Kousidis') Tutorial on syncing videos using reference audio: http://www.dsg-bielefeld.de/dsg_wp/wp-content/uploads/2014/10/video_syncing_fun.pdf
