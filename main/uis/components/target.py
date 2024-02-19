from typing import List, Tuple, Optional
import gradio

import main.globals as globals
import main.uis.globals as uis_globals
import main.uis.temp as temp
from main.utils.filesystem import is_image, is_video


TARGET_FILE : Optional[List[gradio.File]] = [None for _ in range(uis_globals.target_col)]
TARGET_IMAGE : Optional[List[gradio.Image]] = [None for _ in range(uis_globals.target_col)]
TARGET_VIDEO : Optional[List[gradio.Video]] = [None for _ in range(uis_globals.target_col)]
TAB = [None for _ in range(uis_globals.target_col)]


def update(file, i) -> Tuple[gradio.Image, gradio.Video, gradio.File]:
	if file and is_image(file.name):
		temp.target[i] = file.name
		return gradio.Image(value = file.name, visible = True), gradio.Video(value = None, visible = False), gradio.File(value = None, visible = False)
	if file and is_video(file.name):
		temp.target[i] = file.name
		return gradio.Image(value = None, visible = False), gradio.Video(value = file.name, visible = True), gradio.File(value = None, visible = False)
	temp.target = None
	return gradio.Image(value = None, visible = False), gradio.Video(value = None, visible = False), gradio.File(value = None, visible = True, file_types = globals.target_file_types, file_count = 'single',)


def delete_image(i) -> Tuple[gradio.Image, gradio.File]:
    temp.target[i] = None
    return gradio.Image(value = None, visible = False), gradio.File(value = None, visible = True, file_types = globals.target_file_types, file_count = 'single',)


def delete_video(i) -> Tuple[gradio.Video, gradio.File]:
    temp.target[i] = None
    return gradio.Video(value = None, visible = False), gradio.File(value = None, visible = True, file_types = globals.target_file_types, file_count = 'single',)


def update_tab(i) -> None:
    temp.target_tab = i


def render() -> None:
    global TARGET_FILE
    global TARGET_IMAGE
    global TARGET_VIDEO
    global TAB

    for i in range(uis_globals.source_col):
        TAB[i] = gradio.Tab(f'Target {i + 1}')
        with TAB[i]:
            TARGET_FILE[i] = gradio.File(
                file_count = 'single',
                file_types = globals.target_file_types,
            )
            TARGET_IMAGE[i] = gradio.Image(
                visible = False,
                show_label = False,
                interactive = True
            )
            TARGET_VIDEO[i] = gradio.Video(
                visible = False,
                show_label = False,
                interactive = True
        )


def listen() -> None:
    for i in range(uis_globals.source_col):
        TARGET_FILE[i].upload((lambda x, i=i: update(x, i)), inputs = TARGET_FILE[i], outputs = [TARGET_IMAGE[i], TARGET_VIDEO[i], TARGET_FILE[i]])
        TARGET_IMAGE[i].clear((lambda i=i: delete_image(i)), outputs=[TARGET_IMAGE[i], TARGET_FILE[i]])
        TARGET_VIDEO[i].clear((lambda i=i: delete_video(i)), outputs=[TARGET_VIDEO[i], TARGET_FILE[i]])
        TAB[i].select((lambda i=i: update_tab(i)))