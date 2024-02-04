import os

import cv2

import main.globals as globals
from main.utils.filesystem import get_temp_frame_paths, get_temp_directory_path
from main.utils.vision import read_static_image


def smooth_video():
    frame_paths = get_temp_frame_paths(globals.target_path)
    temp_dir = get_temp_directory_path(globals.target_path)
    for i in range(len(frame_paths)):
        if i > 0:
            frame = read_static_image(frame_paths[i])
            frame_prev = read_static_image(frame_paths[i-1])
            if frame.shape == frame_prev.shape:
                averaged_frame = cv2.addWeighted(frame, 0.5, frame_prev, 0.5, 0)
                base_filename, extension = os.path.splitext(os.path.basename(frame_paths[i]))
                averaged_frame_filename = f"{base_filename}_1{extension}"
                averaged_frame_path = os.path.join(temp_dir, averaged_frame_filename)
                
                cv2.imwrite(averaged_frame_path, averaged_frame)