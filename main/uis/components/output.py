from typing import Tuple, Optional
import gradio
import tempfile

import main.globals as globals
import main.utils.wording as wording
import main.uis.temp as temp
from main.core import main_process
from main.utils.normalizer import normalize_output_path
from main.utils.filesystem import clear_temp, is_image, is_video


OUTPUT_IMAGE : Optional[gradio.Image] = None
OUTPUT_VIDEO : Optional[gradio.Video] = None
OUTPUT_START_BUTTON : Optional[gradio.Button] = None
OUTPUT_CLEAR_BUTTON : Optional[gradio.Button] = None


def render() -> None:
	global OUTPUT_IMAGE
	global OUTPUT_VIDEO
	global OUTPUT_START_BUTTON
	global OUTPUT_CLEAR_BUTTON

	with gradio.Column():
		OUTPUT_IMAGE = gradio.Image(
			label = wording.get('output_image_or_video_label'),
			visible = False
		)
		OUTPUT_VIDEO = gradio.Video(
			label = wording.get('output_image_or_video_label')
		)
		with gradio.Row():
			OUTPUT_START_BUTTON = gradio.Button(
				value = wording.get('start_button_label'),
				variant = 'primary',
			)
			OUTPUT_CLEAR_BUTTON = gradio.Button(
				value = wording.get('clear_button_label'),
			)


def listen() -> None:
	output_path_textbox = gradio.Textbox(
		label = wording.get('output_path_textbox_label'),
		value = globals.output_path or tempfile.gettempdir(),
		max_lines = 1
	)
	if output_path_textbox:
		OUTPUT_START_BUTTON.click(start, inputs = output_path_textbox, outputs = [ OUTPUT_IMAGE, OUTPUT_VIDEO ])
	OUTPUT_CLEAR_BUTTON.click(clear, outputs = [ OUTPUT_IMAGE, OUTPUT_VIDEO ])
	


def start(output_path : str) -> Tuple[gradio.Image, gradio.Video]:
	temp_source_path = temp.source[temp.source_tab]
	globals.source_paths = [source_path for source_path in temp_source_path if source_path is not None]
	globals.target_path = temp.target[temp.target_tab]
	globals.output_path = normalize_output_path(globals.source_paths, globals.target_path, output_path)
	if globals.source_paths and globals.target_path:
		main_process()
	if is_image(globals.output_path):
		return gradio.Image(value = globals.output_path, visible = True), gradio.Video(value = None, visible = False)
	if is_video(globals.output_path):
		return gradio.Image(value = None, visible = False), gradio.Video(value = globals.output_path, visible = True)
	return gradio.Image(), gradio.Video()


def clear() -> Tuple[gradio.Image, gradio.Video]:
	if globals.target_path:
		clear_temp(globals.target_path)
	return gradio.Image(value = None), gradio.Video(value = None)
