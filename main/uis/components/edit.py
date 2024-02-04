from typing import Optional, Tuple
import gradio

import main.globals as globals
from main.utils.filesystem import get_temp_frame_paths, is_image, is_video
from main.uis.components import output
import main.uis.temp as temp

CHOICE_FRAME_SLIDER: Optional[gradio.Slider] = None
CHOISE_FRAME_IMAGE: Optional[gradio.Image] = None
EDIT_TAB: Optional[gradio.Tab] = None


def show_frame(frame_number):
    temp_frame_paths = get_temp_frame_paths(globals.target_path)
    if temp_frame_paths:
        frame_path = temp_frame_paths[frame_number - 1]
        return frame_path
    return None


def update_edit() -> Tuple[gradio.Image, gradio.Slider, gradio.Tab]:
    globals.target_path = temp.target[temp.target_tab]
    if is_image(globals.target_path):
        return gradio.Image(visible=False), gradio.Slider(visible=False), gradio.Tab(visible=False)
    if is_video(globals.target_path):
        return gradio.Image(visible=True, value=show_frame(1)), gradio.Slider(visible=True, minimum=1, maximum=len(get_temp_frame_paths(globals.target_path)), label="フレーム番号"), gradio.Tab(visible=True, label="edit")
    return gradio.Image(visible=False), gradio.Slider(visible=False), gradio.Tab(visible=False)


def render():
    global CHOISE_FRAME_IMAGE
    global CHOICE_FRAME_SLIDER 
    global EDIT_TAB

    EDIT_TAB = gradio.Tab(visible=False)
    with EDIT_TAB:
        with gradio.Column():
            CHOISE_FRAME_IMAGE = gradio.Image(visible=True, label="preview")
            CHOICE_FRAME_SLIDER = gradio.Slider(visible=False)


def listen():
    CHOICE_FRAME_SLIDER.change(fn=show_frame, inputs=CHOICE_FRAME_SLIDER, outputs=CHOISE_FRAME_IMAGE)
    output.OUTPUT_START_BUTTON.click(update_edit, outputs=[CHOISE_FRAME_IMAGE, CHOICE_FRAME_SLIDER, EDIT_TAB])