import os
import base64
import time
from fastapi import FastAPI, APIRouter, Body

import main.globals as globals
from main.core import main_process

app = FastAPI()
router = APIRouter()

@router.post("/")
async def process_frames(params = Body(...)) -> dict:
    update_global_variables(params)
    main_process()
    output = to_base64_str(globals.output_path)
    return {"output": output}


def update_global_variables(params):
    for var_name, value in vars(params).items():
        if value is not None:
            if hasattr(globals, var_name):
                setattr(globals, var_name, value)


def to_base64_str(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')


app.include_router(router)


def launch():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)