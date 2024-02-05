import gradio
import sys
sys.path.append('../../')
from main.uis.components import edit, source, target, output, option

def listen():
    output.listen()
    edit.listen()
    source.listen()
    target.listen()
    option.listen()

with gradio.Blocks() as ui:
    with gradio.Row():
        with gradio.Column(scale=4):
            output.render()
            edit.render()
            with gradio.Row():
                with gradio.Blocks():
                    source.render()
                with gradio.Blocks():
                    target.render()
        with gradio.Column(scale=1):
            option.render()

    listen()


    

ui.launch()
