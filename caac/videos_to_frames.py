# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 13:16:02 2025

@author: kpoth
"""

import os
import cv2

def video_frames(videospath, videoframespath):
    for file in os.listdir(videospath):
        count = 0
        file_mod = file[:-4]

        if os.path.exists(f'{videoframespath}/{file_mod}'):
            continue
        print(f"Processing: {file_mod}")
        cap = cv2.VideoCapture(f'{videospath}/{file}') 
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps == 0:
            print(f"Warning: FPS is zero for {file}")
            continue
        duration = frame_count / fps
        print(f"Duration: {duration:.2f} sec")
        sec = 0
        frameRate = 1  
        output_path = os.path.join(videoframespath, file_mod)
        os.makedirs(output_path, exist_ok=True)
        while cap.isOpened():
            if sec > duration:
                break

            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                break

            name = os.path.join(output_path, f'frame{count}.jpg')
            cv2.imwrite(name, frame)
            count += 1
            sec += frameRate
        cap.release()
        
video_path = "/mnt/gs21/scratch/pothugun/COVID_Videos/Videos/"
videoframes_path = "/mnt/gs21/scratch/pothugun/COVID_Videos/Video_Frames/"

video_frames(video_path, videoframes_path)
