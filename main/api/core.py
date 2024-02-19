import os
import base64
import tempfile
import base64

from fastapi import FastAPI, APIRouter, Body
import uvicorn

import main.globals as globals
from main.core import main_process


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


def save_file(file_path: str, encoded_data: str):
    decoded_data = base64.b64decode(encoded_data)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, "wb") as file:
        file.write(decoded_data)


@router.post("/")
async def process_frames(params = Body(...)) -> dict:
    update_global_variables(params)
    sources = params['sources']
    source_paths = []
    for i, source in enumerate(sources):
        source_path = os.path.join(tempfile.mkdtemp(), os.path.basename(f'source{i}.png'))
        save_file(source_path, source)
        source_paths.append(source_path)
    target = params['target']
    target_extension = params['target_extension']
    if target_extension is None or target_extension not in globals.target_file_types:
        raise ValueError("Unsupported file type")
    target_path = os.path.join(tempfile.mkdtemp(), os.path.basename(f'target{target_extension}'))
    save_file(target_path, target)
    globals.source_paths = source_paths
    globals.target_path = target_path
    globals.output_path = os.path.join(tempfile.mkdtemp(), os.path.basename(f'output{target_extension}'))
    main_process()
    output = to_base64_str(globals.output_path)
    return {"output": output}


def launch():
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8000)