# Are your moves any good?

**Hi there! 'Dance Sync', originally called 'Pose' is a program that allows you to compare your dancing to your model dancer's.** <br />

When learning a dance, it can be hard to pinpoint where you're going wrong by just looking in the mirror. Instead, record a video of your dancing and plug it into 'Dance Sync' - it will analyse your movement and compare it to the video of the orginial dancer. <br />

You will get a comparison video, with alerts where you are not in sync and a final score at the end. <br />

![](https://github.com/Mruchus/dance-sync-analysis/blob/f400e40913e5ccd3b6a379e619a5e0c267919b4f/comparisonsample.gif)

[<img src="https://github.com/Mruchus/dance-sync-analysis/blob/main/finalscoresample.png" width="600"/>](https://github.com/Mruchus/dance-sync-analysis/blob/main/finalscoresample.png) <br />

## Quickstart
[Go to the Executing section](#Executing)

Happy dancing! <br />

## How does it work?

### 1. How do we compare the dancers?
We compare the gradients of their limbs i.e. the **angle** of their arms, legs, torso etc... <br />

1. Use a Pose Estimation Model to find the postions of a dancer's joints <br />

2. For the limbs we want to compare, find the corresponding joint postions <br />

3. Compute **gradient** using (x,y) coordinates of joints <br />

[<img src="https://github.com/Mruchus/dance-sync-analysis/blob/6483c792c91f5f57239243693b56d3315a2532a6/gradientEXPLAIN.png" width="600"/>](https://github.com/Mruchus/dance-sync-analysis/blob/6483c792c91f5f57239243693b56d3315a2532a6/gradientEXPLAIN.png) <br />

4. Find percentage difference between the limb's gradient of dancer1 and dancer2. Repeat and average for ALL limbs in once frame. Repeat for all frames <br />

### 2. What happens if we input videos of different lengths?
There may be a case where the dance **song starts a little bit later** in one video. This is not a problem as the program automatically syncs the two dancers so comparisons only happen when they are dancing to the same part of the song! <br />

But how do we do this? <br />
We sync the dancers using the music! Both dancers are dancing to the same music so we can use this to find which when they are doing the same choreo. <br />

1. Find the offset between the audio in the two videos by... <br />
- Computing the cross-correlation of the sound waves (function to tell us how similar the waves are,) for different time lags <br />
(Good videos explaining cross-correlations here: https://youtu.be/_r_fDlM0Dx0 | https://youtu.be/ngEC3sXeUb4 ) <br />
- Find the time lag (a.k.a. **phase difference**) where the cross-correlation is a maximum <br />
(This is also known as **TIME DELAY ANALYSIS**)

![](https://github.com/Mruchus/dance-sync-analysis/blob/65e7469d1b7e8d42438a568b58fddfc50054b2ec/syncExplain1.JPG)

2. Trim the video in which the dancing starts later <br />
- We cut the video with same duration as the time lag so... <br />
- both dancers start the same choreography at the beginning of the video! <br />

![](https://github.com/Mruchus/dance-sync-analysis/blob/65e7469d1b7e8d42438a568b58fddfc50054b2ec/syncExplain2.JPG)

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
- Download the folder <br />
- Launch with these arguments: **python dance.py video/chuu.mov video/cyves.mov** <br />
- Make sure the videos are in **.mov** format and are the same dimension i.e. 1280x720 px <br />

## Acknowledgments
* Inspiration: https://www.youtube.com/watch?v=zxhXj72WClE
* Useful documentation of how pose estimation works: https://google.github.io/mediapipe/solutions/pose.html
* (Dr Spyros Kousidis') Tutorial on syncing videos using reference audio: http://www.dsg-bielefeld.de/dsg_wp/wp-content/uploads/2014/10/video_syncing_fun.pdf
