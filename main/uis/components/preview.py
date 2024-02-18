from typing import Optional, Tuple
import gradio

import main.globals as globals
import main.uis.temp as temp
import main.uis.components.output as output
from main.process.core import process_preview
from main.utils.filesystem import get_temp_frame_paths, is_video, is_image
from main.utils.ffmpeg import extract_frames
from main.utils.vision import detect_video_resolution, pack_resolution

CHOICE_FRAME_SLIDER: Optional[gradio.Slider] = None
CHOISE_FRAME_IMAGE: Optional[gradio.Image] = None
PREVIEW_BUTTON: Optional[gradio.Button] = None


def show_frame(frame_number):
    temp_frame_paths = get_temp_frame_paths(globals.target_path)
    if temp_frame_paths:
        frame_path = temp_frame_paths[frame_number - 1]
        return frame_path
    return None


def show_preview():
    temp_source_path = temp.source[temp.source_tab]
    globals.source_paths = [source_path for source_path in temp_source_path if source_path is not None]
    globals.target_path = temp.target[temp.target_tab]
    if globals.source_paths and globals.target_path:
        if is_image(globals.target_path):
            return process_preview(globals.target_path)
        elif is_video(globals.target_path):
            target_video_resolution = detect_video_resolution(globals.target_path)
            output_video_resolution = pack_resolution(target_video_resolution)
            extract_frames(globals.target_path, output_video_resolution, globals.output_video_fps)
            temp_frame_paths = get_temp_frame_paths(globals.target_path)
            if temp_frame_paths:
                return process_preview(temp_frame_paths[0])
    return None


def preview() -> gradio.Image:
    globals.target_path = temp.target[temp.target_tab]
    if is_video(globals.target_path):
        return gradio.Image(visible=True, value=show_preview())
    elif is_image(globals.target_path):
        return gradio.Image(visible=True, value=show_preview())
    return gradio.Image(visible=True)
    

def show_slider() -> gradio.Slider:
    temp_frame_paths = get_temp_frame_paths(globals.target_path)
    if temp_frame_paths:
        return gradio.Slider(visible=True, minimum=1, maximum=len(temp_frame_paths), step=1, label="frame number")
    return gradio.Slider(visible=False)


def render():
    global CHOISE_FRAME_IMAGE
    global CHOICE_FRAME_SLIDER 
    global PREVIEW_BUTTON

    with gradio.Column():
        CHOISE_FRAME_IMAGE = gradio.Image(visible=True, label="preview")
        CHOICE_FRAME_SLIDER = gradio.Slider(visible=False)
        PREVIEW_BUTTON = gradio.Button(value="Preview", variant = 'primary',)


def listen():
    CHOICE_FRAME_SLIDER.change(fn=show_frame, inputs=CHOICE_FRAME_SLIDER, outputs=CHOISE_FRAME_IMAGE)
    PREVIEW_BUTTON.click(fn=preview, outputs=CHOISE_FRAME_IMAGE)
    output.OUTPUT_START_BUTTON.click(fn=show_slider, outputs=CHOICE_FRAME_SLIDER)