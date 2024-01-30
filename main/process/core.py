import os

os.environ['OMP_NUM_THREADS'] = '1'
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import List
from logging import INFO
import time

from tqdm import tqdm

import main.globals as globals
import main.instances as instances
import main.face_store as face_store
from main.types import ProcessFrames, UpdateProcess
from main.face_modules.swap_face import SwapFace
from main.utils.vision import read_image, read_static_image, read_static_images, write_image
from main.utils.filesystem import is_video, get_temp_frame_paths, create_temp, move_temp, clear_temp
from main.utils.ffmpeg import extract_frames, merge_video, restore_audio
import main.utils.logger as logger
import main.utils.wording as wording


def multi_process_frames(source_paths : List[str], temp_frame_paths : List[str], process_frames : ProcessFrames) -> None:
	with tqdm(total = len(temp_frame_paths), desc = 'processing', unit = 'frame', ascii = ' =', disable = INFO in [ 'warn', 'error' ]) as progress:
		progress.set_postfix(
		{
			'execution_providers': encode_execution_providers(globals.device),
			'execution_thread_count': globals.thread,
			'execution_queue_count': globals.queue
		})
		with ThreadPoolExecutor(max_workers = globals.thread) as executor:
			futures = []
			queue_frame_paths : Queue[str] = create_queue(temp_frame_paths)
			queue_per_future = max(len(temp_frame_paths) // globals.thread * globals.queue, 1)
			while not queue_frame_paths.empty():
				submit_frame_paths = pick_queue(queue_frame_paths, queue_per_future)
				future = executor.submit(process_frames, source_paths, submit_frame_paths, progress.update)
				futures.append(future)
			for future_done in as_completed(futures):
				future_done.result()


def create_queue(temp_frame_paths : List[str]) -> Queue[str]:
	queue : Queue[str] = Queue()
	for frame_path in temp_frame_paths:
		queue.put(frame_path)
	return queue


def pick_queue(queue : Queue[str], queue_per_future : int) -> List[str]:
	queues = []
	for _ in range(queue_per_future):
		if not queue.empty():
			queues.append(queue.get())
	return queues


def encode_execution_providers(execution_providers : List[str]) -> List[str]:
	return [ execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers ]


def process_frames(source_paths: List[str], temp_frame_paths: List[str], update_progress: UpdateProcess) -> None:
	source_frames = read_static_images(source_paths)
	swapper = SwapFace()
	for temp_frame_path in temp_frame_paths:
		temp_frame = read_image(temp_frame_path)
		result_frame = swapper.swap(source_frames=source_frames, target_frame=temp_frame)
		write_image(temp_frame_path, result_frame)
		update_progress()


def process_image(source_paths : List[str], target_path : str, output_path : str) -> None:
	print(source_paths)
	print(target_path)
	print(output_path)
	swapper = SwapFace()
	source_frames = read_static_images(source_paths)
	target_frame = read_static_image(target_path)
	result_frame = swapper.swap(source_frames=source_frames, target_frame=target_frame)
	write_image(output_path, result_frame)


def post_process() -> None:
	instances.reset_instances()
	face_store.reset_face_store()


def process_video(start_time : float) -> None:
	clear_temp(globals.target_path)
	logger.info(wording.get('creating_temp'), __name__.upper())
	create_temp(globals.target_path)
	logger.info(wording.get('extracting_frames_fps').format(video_fps = globals.output_video_fps), __name__.upper())
	extract_frames(globals.target_path, globals.output_video_resolution, globals.output_video_fps)
	temp_frame_paths = get_temp_frame_paths(globals.target_path)
	if temp_frame_paths:
		logger.info(wording.get('processing'), globals.process_mode)
		multi_process_frames(globals.source_paths, temp_frame_paths, process_frames)
		post_process()
	else:
		logger.error(wording.get('temp_frames_not_found'), __name__.upper())
		return
	logger.info(wording.get('merging_video_fps').format(video_fps = globals.output_video_fps), __name__.upper())
	if not merge_video(globals.target_path, globals.output_video_fps):
		logger.error(wording.get('merging_video_failed'), __name__.upper())
		return
	logger.info(wording.get('restoring_audio'), __name__.upper())
	if not restore_audio(globals.target_path, globals.output_path, globals.output_video_fps):
		logger.warn(wording.get('restoring_audio_skipped'), __name__.upper())
		move_temp(globals.target_path, globals.output_path)
	logger.info(wording.get('clearing_temp'), __name__.upper())
	clear_temp(globals.target_path)
	if is_video(globals.output_path):
		seconds = '{:.2f}'.format((time.time() - start_time))
		logger.info(wording.get('processing_video_succeed').format(seconds = seconds), __name__.upper())
	else:
		logger.error(wording.get('processing_video_failed'), __name__.upper())