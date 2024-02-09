import sys

sys.path.append('../')
import main.globals as globals
from main.process.core import process_video
from main.utils.filesystem import resolve_relative_path
from main.utils.vision import detect_video_resolution, pack_resolution
import main.utils.logger as logger

def test_process_video():
    logger.init(globals.log_level)
    globals.source_paths = ['/Users/haruki/Sites/dev/ml/Mimix/tests/images/hou-1.JPG', '/Users/haruki/Sites/dev/ml/Mimix/tests/images/hou-2.JPG', '/Users/haruki/Sites/dev/ml/Mimix/tests/images/hou-3.JPG', '/Users/haruki/Sites/dev/ml/Mimix/tests/images/hou-4.JPG']
    globals.target_path = '/Users/haruki/Sites/dev/ml/Mimix/tests/images/demo-2.MP4'
    globals.output_path = '/Users/haruki/Sites/dev/ml/Mimix/tests/images/result.mp4'

    print(globals.source_paths)
    print(globals.target_path)
    print(globals.output_path)
    
    process_video(0)

test_process_video()