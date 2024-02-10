from argparse import ArgumentParser, HelpFormatter

import main.globals as globals
import main.uis.core as ui
import main.api.core as api

from main.utils.filesystem import resolve_relative_path
from main.utils.download import conditional_download, is_download_done


def pre_check() -> bool:
    model_urls = [
        'https://github.com/tamoharu/assets/releases/download/models/arcface_w600k_r50.onnx',
        'https://github.com/tamoharu/assets/releases/download/models/codeformer.onnx',
        'https://github.com/tamoharu/assets/releases/download/models/face_occluder.onnx',
        'https://github.com/tamoharu/assets/releases/download/models/face_parser.onnx',
        'https://github.com/tamoharu/assets/releases/download/models/inswapper_128.onnx',
        'https://github.com/tamoharu/assets/releases/download/models/yolov8n-face-dynamic.onnx'
    ]
    download_directory_path = resolve_relative_path('./models')
    conditional_download(download_directory_path, model_urls)
    return True


def run():
    # pre_check()
    program = ArgumentParser(formatter_class = lambda prog: HelpFormatter(prog, max_help_position = 120), add_help = False)
    program.add_argument('-api', help='Run in API mode', action='store_true', dest='api_mode')
    program.add_argument('-webcam', help='Run in Webcam mode', action='store_true', dest='webcam_mode')
    args = program.parse_args()
    if args.api_mode:
        api.launch()
    elif args.webcam_mode:
        globals.webcam = True
    else:
        ui.launch()


if __name__ == '__main__':
    run()