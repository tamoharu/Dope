from typing import Optional

import gradio


ABOUT_BUTTON: Optional[gradio.Button] = None


def render():
    global ABOUT_BUTTON
    
    ABOUT_BUTTON = gradio.Button(
        value = 'Mimix',
        variant = 'primary',
    )