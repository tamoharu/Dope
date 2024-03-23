import os

os.environ['OMP_NUM_THREADS'] = '1'
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import List
from logging import INFO
import time
import shutil

from tqdm import tqdm

import main.globals as globals
import main.instances as instances
import main.face_store as face_store
import main.utils.logger as logger
import main.utils.wording as wording
from main.type import ProcessFrames, UpdateProcess, Frame
from main.face_modules.swap_face import swap_face, create_source_embedding
from main.utils.vision import read_image, read_static_image, read_static_images, write_image, detect_video_resolution, pack_resolution, detect_video_fps
from main.utils.filesystem import is_video, get_temp_frame_paths, create_temp, move_temp, clear_temp, is_image
from main.utils.ffmpeg import extract_frames, merge_video, restore_audio, compress_image
from main.face_modules.smooth_video import smooth_video


def multi_process_frames(source_paths : List[str], temp_frame_paths : List[str], process_frames : ProcessFrames) -> None:
	with tqdm(total = len(temp_frame_paths), desc = 'processing', unit = 'frame', ascii = ' =', disable = INFO in [ 'warn', 'error' ]) as progress:
		progress.set_postfix(
		{
			'device': globals.device,
			'thread': globals.thread,
			'queue': globals.queue
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


def clear() -> None:
	instances.clear_instances()
	face_store.reset_face_store()


def process_frames(source_paths: List[str], temp_frame_paths: List[str], update_progress: UpdateProcess) -> None:
	source_embedding = face_store.source_embedding
	if globals.process_mode == 'swap':
		for temp_frame_path in temp_frame_paths:
			temp_frame = read_image(temp_frame_path)
			result_frame = swap_face(source_embedding, temp_frame)
			write_image(temp_frame_path, result_frame)
			update_progress()


def process_preview(target_path: str) -> Frame:
	if globals.process_mode == 'swap':
		source_frames = read_static_images(globals.source_paths)
		target_frame = read_static_image(target_path)
		create_source_embedding(source_frames)
		source_embedding = face_store.source_embedding
		result_frame = swap_face(source_embedding, target_frame)
		result_frame = result_frame[:, :, ::-1]
		return result_frame
	return None


def process_image(start_time : float) -> None:
	clear_temp(globals.target_path)
	clear()
	shutil.copy2(globals.target_path, globals.output_path)
	if globals.process_mode == 'swap':
		source_frames = read_static_images(globals.source_paths)
		target_frame = read_static_image(globals.target_path)
		create_source_embedding(source_frames)
		source_embedding = face_store.source_embedding
		result_frame = swap_face(source_embedding ,target_frame)
		write_image(globals.output_path, result_frame)
	logger.info(wording.get('compressing_image'), __name__.upper())
	if not compress_image(globals.output_path):
		logger.error(wording.get('compressing_image_failed'), __name__.upper())
	if is_image(globals.output_path):
		seconds = '{:.2f}'.format((time.time() - start_time) % 60)
		logger.info(wording.get('processing_image_succeed').format(seconds = seconds), __name__.upper())
	else:
		logger.error(wording.get('processing_image_failed'), __name__.upper())


def process_video(start_time : float) -> None:
	logger.info(wording.get('creating_temp'), __name__.upper())
	print(f'globals.device: {globals.device}')
	create_temp(globals.target_path)
	clear()
	if globals.keep_fps:
		globals.output_video_fps = detect_video_fps(globals.target_path)
	logger.info(wording.get('extracting_frames_fps').format(video_fps = globals.output_video_fps), __name__.upper())
	target_video_resolution = detect_video_resolution(globals.target_path)
	output_video_resolution = pack_resolution(target_video_resolution)
	extract_frames(globals.target_path, output_video_resolution, globals.output_video_fps)
	temp_frame_paths = get_temp_frame_paths(globals.target_path)
	if temp_frame_paths:
		logger.info(wording.get('processing'), globals.process_mode)
		source_frames = read_static_images(globals.source_paths)
		create_source_embedding(source_frames)
		multi_process_frames(globals.source_paths, temp_frame_paths, process_frames)
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
	if is_video(globals.output_path):
		seconds = '{:.2f}'.format((time.time() - start_time))
		logger.info(wording.get('processing_video_succeed').format(seconds = seconds), __name__.upper())
	else:
		logger.error(wording.get('processing_video_failed'), __name__.upper())