import time

import main.globals as globals
from main.utils.filesystem import is_image, is_video
from main.process.core import process_image, process_video

def main_process():
    start_time = time.time()
    if is_image(globals.target_path):
        process_image(start_time)
    if is_video(globals.target_path):
        process_video(start_time)