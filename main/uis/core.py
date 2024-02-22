import gradio
import sys
sys.path.append('../../')
from main.uis.components import about, preview, source, target, output, option


def listen():
    output.listen()
    preview.listen()
    source.listen()
    target.listen()
    option.listen()


def render() -> gradio.Blocks:
    with gradio.Blocks() as ui:
        with gradio.Row():
            with gradio.Column(scale=4):
                with gradio.Blocks():
                    with gradio.Row():
                        output.render()
                        preview.render()
                with gradio.Row():
                    with gradio.Blocks():
                        source.render()
                    with gradio.Blocks():
                        target.render()
            with gradio.Column(scale=1):
                option.render()
        listen()
    return ui


def launch():
    render().launch()