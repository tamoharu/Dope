import main.globals as globals
import main.uis.globals as uis_globals

source = [[None for _ in range(uis_globals.source_row)] for _ in range(uis_globals.source_col)]
source_tab = 0

target = [None for _ in range(uis_globals.target_col)]
target_tab = 0

process_mode = globals.process_mode
thread = globals.thread
queue = globals.queue
device = globals.device
keep_fps = globals.keep_fps
detect_face_model = globals.detect_face_model
score_threshold = globals.score_threshold
iou_threshold = globals.iou_threshold
enhance_face_model = globals.enhance_face_model
swap_face_model = globals.swap_face_model
blend_strength = globals.blend_strength