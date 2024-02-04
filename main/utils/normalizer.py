from typing import List, Optional
import os

from main.utils.filesystem import is_file, is_directory
from main.type import Fps


def normalize_output_path(source_paths : List[str], target_path : str, output_path : str) -> Optional[str]:
	if is_file(target_path) and is_directory(output_path):
		target_name, target_extension = os.path.splitext(os.path.basename(target_path))
		if source_paths and is_file(source_paths[0]):
			source_name, _ = os.path.splitext(os.path.basename(source_paths[0]))
			return os.path.join(output_path, source_name + '-' + target_name + target_extension)
		return os.path.join(output_path, target_name + target_extension)
	if is_file(target_path) and output_path:
		_, target_extension = os.path.splitext(os.path.basename(target_path))
		output_name, output_extension = os.path.splitext(os.path.basename(output_path))
		output_directory_path = os.path.dirname(output_path)
		if is_directory(output_directory_path) and output_extension:
			return os.path.join(output_directory_path, output_name + target_extension)
		return None
	return output_path


def normalize_fps(fps : Optional[float]) -> Optional[Fps]:
	if fps is not None:
		if fps < 1.0:
			return 1.0
		if fps > 60.0:
			return 60.0
		return fps
	return None
