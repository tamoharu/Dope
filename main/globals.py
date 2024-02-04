from typing import List, Optional
from main.type import DetectFaceModel, SwapFaceModel, MaskFaceModel, MaskFaceRegion, EnhanceFaceModel, Process, TempFrameFormat, OutputVideoEncoder, OutputVideoPreset, LogLevel

source_paths : List[str] = None
target_path : str = None
output_path : str = None
thread: int = 32
queue: int = 4
device = ['CPUExecutionProvider']
score_threshold: float = 0.35
iou_threshold: float = 0.4
detect_face_model: DetectFaceModel  = 'yolov8'
mask_face_model: List[MaskFaceModel] = ['face_occluder']
mask_face_regions: List[MaskFaceRegion] = ['right-eye', 'left-eye']
enhance_face_model: EnhanceFaceModel = 'codeformer'
swap_face_model: SwapFaceModel  = 'inswapper'
process_mode: Process = 'swap'

blend_strangth: int = 100

trim_frame_start : int = None
trim_frame_end : int = None
temp_frame_format : TempFrameFormat = 'jpg'
temp_frame_quality : int = 100
keep_temp : bool = None

output_image_quality : int = 80
output_video_encoder : OutputVideoEncoder = 'libx264'
output_video_preset : OutputVideoPreset = 'veryfast'
output_video_quality : int = 80
output_video_fps : float = 25

log_level : LogLevel = 'info'