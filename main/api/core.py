import os
import base64
import tempfile
import base64

from fastapi import FastAPI, APIRouter, Body
import numpy as np
import cv2
import uvicorn

import main.globals as globals
from main.core import main_process
from main.utils.vision import write_image


app = FastAPI()
router = APIRouter()


def update_global_variables(params):
    for var_name, value in params.items():
        if value is not None:
            if hasattr(globals, var_name):
                setattr(globals, var_name, value)


def to_base64_str(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')
    

def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    nparr = np.fromstring(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def determine_file_type(decoded_data):
    file_signatures = {
        b'\x89PNG\r\n\x1a\n': 'png',
        b'\xff\xd8\xff': 'jpg',
        b'RIFF....WEBPVP': 'webp',
        b'\x00\x00\x00\x18ftypmp42': 'mp4',
        b'\x00\x00\x00\x20ftypisom': 'mp4',
        b'\x00\x00\x00\x14ftypqt  ': 'mov',
    }
    file_header = decoded_data[:16]
    for signature, format in file_signatures.items():
        if file_header.startswith(signature) or (signature in file_header):
            return format
    return None


@router.post("/")
async def process_frames(params = Body(...)) -> dict:
    update_global_variables(params)
    sources = params['sources']
    source_paths = []
    for i, source in enumerate(sources):
        source_path = os.path.join(tempfile.mkdtemp(), os.path.basename(f'source{i}.png'))
        source_frame = base64_to_image(source)
        write_image(source_path, source_frame)
        source_paths.append(source_path)
    target = params['target']
    decoded_target = base64.b64decode(target)
    target_extension = determine_file_type(decoded_target)
    if target_extension is None:
        raise ValueError("Unsupported file type")
    target_path = os.path.join(tempfile.mkdtemp(), os.path.basename(f'target.{target_extension}'))
    target_frame = base64_to_image(target)
    write_image(target_path, target_frame)
    globals.source_paths = source_paths
    globals.target_path = target_path
    globals.output_path = os.path.join(tempfile.mkdtemp(), os.path.basename(f'output.{target_extension}'))
    main_process()
    output = to_base64_str(globals.output_path)
    return {"output": output}


app.include_router(router)


def launch():
    uvicorn.run(app, host="0.0.0.0", port=8000)