import gradio
import os
import sys
sys.path.append('../../')
from typing import Dict, List, Optional
from types import ModuleType
from main.uis.components import edit, source, target, output
from main.uis.type import Component, ComponentName

UI_COMPONENTS: Dict[ComponentName, Component] = {}
UI_LAYOUT_MODULES : List[ModuleType] = []

def listen():
    output.listen()
    edit.listen()
    source.listen()
    target.listen()

with gradio.Blocks() as ui:
    output.render()
    edit.render()
    with gradio.Row():
        with gradio.Blocks():
            source.render()
        with gradio.Blocks():
            target.render()

    listen()


def get_ui_component(name : ComponentName) -> Optional[Component]:
	if name in UI_COMPONENTS:
		return UI_COMPONENTS[name]
	return None


def register_ui_component(name : ComponentName, component: Component) -> None:
	UI_COMPONENTS[name] = component


    

ui.launch()
