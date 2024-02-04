import gradio
import os
import tempfile
import numpy as np

from main.utils.vision import write_image
import main.uis.temp as temp
import main.uis.globals as uis_globals


SOURCE_IMAGE = [[None for _ in range(uis_globals.source_row)] for _ in range(uis_globals.source_col)]
TAB = [None for _ in range(uis_globals.source_col)]


def add(image, i, j) -> None:
    if image is not None:
        image = np.array(image)
        temp_dir = tempfile.mkdtemp()
        temp_source_path = os.path.join(temp_dir, os.path.basename(f'{i}-{j}.png'))
        write_image(temp_source_path, image)
        temp.source[i][j] = temp_source_path


def delete(i, j) -> None:
    temp_source_path = temp.source[i][j]
    if temp_source_path and os.path.exists(temp_source_path):
        os.remove(temp_source_path)
        temp.source[i][j] = None


def update_tab(i) -> None:
    temp.source_tab = i


def render() -> None:
    global SOURCE_IMAGE
    global TAB
    for i in range(uis_globals.source_col):
        TAB[i] = gradio.Tab(f'Source {i + 1}')
        with TAB[i]:
            with gradio.Row(variant='panel'):
                for j in range(uis_globals.source_row):
                    SOURCE_IMAGE[i][j] = gradio.Image(interactive=True)


def listen() -> None:
    for i in range(uis_globals.source_col):
        TAB[i].select((lambda i=i: update_tab(i)))
        for j in range(uis_globals.source_row):
            SOURCE_IMAGE[i][j].change((lambda x, i=i, j=j: add(x, i, j)), inputs = [SOURCE_IMAGE[i][j]])
            SOURCE_IMAGE[i][j].clear((lambda i=i, j=j: delete(i, j)))