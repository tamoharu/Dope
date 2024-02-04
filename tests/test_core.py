import sys
sys.path.append('../')
import main.globals as globals
from main.core import main_process

def test_main_process():
    globals.source_paths = ['/Users/haruki/Sites/dev/ml/Dope/tests/images/hou-1.JPG', '/Users/haruki/Sites/dev/ml/Dope/tests/images/hou-2.JPG', '/Users/haruki/Sites/dev/ml/Dope/tests/images/hou-3.JPG', '/Users/haruki/Sites/dev/ml/Dope/tests/images/hou-4.JPG']
    globals.target_path = '/Users/haruki/Sites/dev/ml/Dope/tests/images/manyface.jpg'
    globals.output_path = '/Users/haruki/Sites/dev/ml/Dope/tests/images/result.jpg'
    main_process()

test_main_process()