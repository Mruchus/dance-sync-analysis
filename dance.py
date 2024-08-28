import os
import subprocess
import math
from enum import Enum
from statistics import mean
from typing import List, Tuple

import cv2
import numpy as np
import mediapipe as mp

OUTPUT_DIR = "output"
EXIST_FLAG = "-n"  # ignore existing file, change to -y to always overwrite
PRAAT_PATH = "/Applications/Praat.app/Contents/MacOS/Praat"
SEARCH_INTERVAL = 30  # in secs
FPS = 24.0
SYNC_THRESHOLD = 0.15 # would allow for 180 * 0.15 = 27 degrees off

# MediaPipe setup
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class PoseLandmark(Enum):
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


LIMB_CONNECTIONS = [
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
]


def get_video_duration(filename: str) -> float:
    """Returns the duration of a video clip in seconds."""
    video = cv2.VideoCapture(filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return frame_count / fps


def get_frame_count(filename: str) -> Tuple[cv2.VideoCapture, int]:
    """Returns a tuple of (captured video, frame count)."""
    video = cv2.VideoCapture(filename)
    frame_count = int(math.floor(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    return video, frame_count


def extract_landmarks(video_path: str) -> Tuple[List[List[Tuple[float, float]]], List[np.ndarray], List]:
    """Extracts pose landmarks from a video."""
    pose = mp_pose.Pose()
    xy_landmark_coords = []
    frames = []
    landmarks = []

    video, frame_count = get_frame_count(video_path)

    for _ in range(frame_count):
        success, image = video.read()
        if not success:
            break
        frames.append(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(image_rgb)
        landmarks.append(results)

        if results.pose_landmarks:
            xy_landmark_coords.append([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])
        else:
            xy_landmark_coords.append([(0, 0)] * len(PoseLandmark))

    return xy_landmark_coords, frames, landmarks


def calculate_limb_angles(frame_landmarks: List[List[Tuple[float, float]]]) -> List[List[float]]:
    """Calculates limb angles for each frame."""
    frame_angles = []

    for landmarks in frame_landmarks:
        limb_angles = []
        for start, end in LIMB_CONNECTIONS:
            try:
                start_point = np.array(landmarks[start.value])
                end_point = np.array(landmarks[end.value])

                # calculate angle of limb with respect to vertical (y-axis)
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                # arctan2 identifies sign (i.e. quadrant) + deals with zero values
                angle = np.degrees(np.arctan2(dx, dy))

                # normalize angle -> between 0 and 180 degrees
                angle = abs(angle)
                if angle > 180:
                    angle = 360 - angle

                limb_angles.append(angle)
            except (IndexError, ZeroDivisionError):
                # fallback if zero division error
                # shouldn't really happen cus arctan2 deals with this
                limb_angles.append(0)
        frame_angles.append(limb_angles)

    return frame_angles


def compare_dancers(ref_landmarks: List[List[Tuple[float, float]]],
                    comp_landmarks: List[List[Tuple[float, float]]],
                    ref_frames: List[np.ndarray],
                    comp_frames: List[np.ndarray],
                    ref_pose_results: List,
                    comp_pose_results: List) -> float:
    """Compares two dancers and returns a synchronization score."""
    # get number of comparable frames
    num_frames = min(len(ref_landmarks), len(comp_landmarks))

    ref_angles = calculate_limb_angles(ref_landmarks)
    comp_angles = calculate_limb_angles(comp_landmarks)

    # keep track of current number of out of sync frames (OFS)
    out_of_sync_frames = 0
    score = 100.0

    print("Analysing dancers...")
    video_writer = cv2.VideoWriter(f'{OUTPUT_DIR}/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (2 * 720, 1280))

    for frame_idx in range(num_frames):
        # difference in angle for each limb
        frame_diffs = [abs(ref_angles[frame_idx][j] - comp_angles[frame_idx][j]) / 180 for j in
                       range(len(LIMB_CONNECTIONS))]
        frame_diff = mean(frame_diffs)

        ref_frame = ref_frames[frame_idx]
        comp_frame = comp_frames[frame_idx]

        # annotation skeleton and score on the frame
        mp_draw.draw_landmarks(ref_frame, ref_pose_results[frame_idx].pose_landmarks, mp_pose.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(comp_frame, comp_pose_results[frame_idx].pose_landmarks, mp_pose.POSE_CONNECTIONS)

        display = np.concatenate((ref_frame, comp_frame), axis=1)

        color = (0, 0, 255) if frame_diff > SYNC_THRESHOLD else (255, 0, 0)

        cv2.putText(display, f"Diff: {frame_diff:.2f}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # determine if synced
        if frame_diff > SYNC_THRESHOLD:
            out_of_sync_frames += 1

        score = ((frame_idx + 1 - out_of_sync_frames) / (frame_idx + 1)) * 100.0
        cv2.putText(display, f"Score: {score:.2f}%", (ref_frame.shape[1] + 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                    3)

        cv2.imshow(str(frame_idx), display)
        video_writer.write(display)
        cv2.waitKey(1)

    video_writer.release()
    return score


def convert_to_same_framerate(clip: str) -> str:
    """Converts a video clip to 24 fps and returns the path to the converted clip."""
    clip_name = os.path.splitext(os.path.basename(clip))[0]
    clip_24 = f"{OUTPUT_DIR}/{clip_name}_24.mov"
    os.system(f"ffmpeg {EXIST_FLAG} -i {clip} -filter:v fps={FPS} {clip_24}")
    return clip_24


def validate_reference_clip(ref_clip: str, comparison_clip: str):
    """Validates that the reference clip is longer than the comparison clip."""
    _, ref_frame_count = get_frame_count(ref_clip)
    _, comp_frame_count = get_frame_count(comparison_clip)
    if ref_frame_count <= comp_frame_count:
        raise ValueError(f"Reference clip {ref_clip} must be longer than comparison clip {comparison_clip}")


def convert_to_wav(clip: str) -> str:
    """Converts a video clip to WAV format and returns the path to the WAV file."""
    clip_name = os.path.splitext(os.path.basename(clip))[0]
    clip_wav = f"{OUTPUT_DIR}/{clip_name}.wav"
    os.system(f"ffmpeg {EXIST_FLAG} -i {clip} {clip_wav}")
    return clip_wav


def find_sound_offset(ref_wav: str, comparison_wav: str) -> float:
    """Finds the offset between two WAV files using Praat."""
    command = f"{PRAAT_PATH} --run 'crosscorrelate.praat' {ref_wav} {comparison_wav} 0 {SEARCH_INTERVAL}"
    # note: code in separate praat file
    offset = subprocess.check_output(command, shell=True)
    # (did some formatting here to get the offset from b'0.23464366914074475\n' to 0.23464366914074475)
    return abs(float(str(offset)[2:-3]))


def trim_clips(ref_clip: str, comparison_clip: str, offset: float) -> Tuple[str, str]:
    """Trims both clips to the same duration based on the calculated offset."""
    duration = get_video_duration(comparison_clip)

    ref_name = os.path.splitext(os.path.basename(ref_clip))[0]
    comp_name = os.path.splitext(os.path.basename(comparison_clip))[0]

    ref_cut = f"{OUTPUT_DIR}/{ref_name}_cut.mov"
    comp_cut = f"{OUTPUT_DIR}/{comp_name}_cut.mov"

    os.system(f"ffmpeg {EXIST_FLAG} -i {ref_clip} -ss {offset} -t {duration} {ref_cut}")
    os.system(f"ffmpeg {EXIST_FLAG} -i {comparison_clip} -ss 0 -t {duration} {comp_cut}")

    return ref_cut, comp_cut


def main(ref_clip: str, comparison_clip: str, compare_only: bool = False):
    # ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not compare_only:
        ref_clip_24 = convert_to_same_framerate(ref_clip)
        comp_clip_24 = convert_to_same_framerate(comparison_clip)


        validate_reference_clip(ref_clip, comparison_clip)

        ref_wav = convert_to_wav(ref_clip)
        comp_wav = convert_to_wav(comparison_clip)

        offset = find_sound_offset(ref_wav, comp_wav)

        # trip clips so aligned based on detected offset
        ref_cut, comp_cut = trim_clips(ref_clip_24, comp_clip_24, offset)
    else:
        # if `compare_only` is True -> clips already trimmed and synchronised
        ref_cut, comp_cut = ref_clip, comparison_clip

    print(f"Processing reference: {ref_cut}, comparison: {comp_cut}")

    # extract body landmarks, frames and pose results
    ref_landmarks, ref_frames, ref_pose_results = extract_landmarks(ref_cut)
    comp_landmarks, comp_frames, comp_pose_results = extract_landmarks(comp_cut)

    # compare
    score = compare_dancers(ref_landmarks, comp_landmarks, ref_frames, comp_frames, ref_pose_results, comp_pose_results)

    print(f"\nYou are {score:.2f}% in sync with your model dancer!")


if __name__ == "__main__":
    import sys

    # correct number of args provided?
    if len(sys.argv) < 3:
        print(f"Usage:\n {sys.argv[0]} <ref_clip> <comparison_clip> [--compare-only]")
        sys.exit(-1)

    # parsing the arguments
    ref_clip = sys.argv[1]
    comparison_clip = sys.argv[2]
    compare_only = len(sys.argv) > 3 and sys.argv[3] == '--compare-only'

    main(ref_clip, comparison_clip, compare_only)
