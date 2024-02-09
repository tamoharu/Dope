from argparse import ArgumentParser, HelpFormatter

import main.globals as globals
import main.uis.core as ui
import main.api.core as api

def run():
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