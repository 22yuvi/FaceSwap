import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib
from pathlib import Path
from typing import List, Optional, Any, Callable
from tqdm import tqdm
import onnxruntime
# import tensorflow
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
import signal
import importlib
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import cv2
import threading
import insightface
import numpy
from insightface.app.common import Face
# from gfpgan.utils import GFPGANer
import streamlit as st


Face = Face
Frame = numpy.ndarray[Any, Any]
FACE_ENHANCER = None
FACE_ANALYSER = None
FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()
output_video_encoder = 'libx264'

working_dir = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(working_dir,"source.png")
target_path = os.path.join(working_dir,"target.mp4")
source_name, _ = os.path.splitext(os.path.basename(source_path))
target_name, target_extension = os.path.splitext(os.path.basename(target_path))
output_path = os.path.join(working_dir, source_name + '-' + target_name + target_extension)

face_swapper_path = working_dir
face_enhancer_path = working_dir

def pre_check() -> bool:
    if sys.version_info < (3, 9):
        print('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        print('ffmpeg is not installed.')
        return False
    return True

def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))

def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False

def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False

def get_face_analyser() -> Any:
    global FACE_ANALYSER
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        return get_face_analyser().get(frame)
    except ValueError:
        return None

def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None

def pre_start() -> bool:
    if not is_image(source_path):
        st.write('Select an image for source path.')
        return False
    elif not get_one_face(cv2.imread(source_path)):
        st.write('No face in source path detected.')
        return False
    if  not is_video(target_path):
        st.write('Select a video for target path.')
        return False
    return True

def detect_fps(target_path: str) -> float:
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', target_path]
    output = subprocess.check_output(command).decode().strip().split('/')
    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        pass
    return 30

def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
    commands.extend(args)
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception:
        pass
    return False

def extract_frames(target_path: str, temp_directory_path: str, fps: float = 30) -> bool:
    temp_frame_quality = 0
    return run_ffmpeg(['-hwaccel', 'auto', '-i', target_path, '-q:v', str(temp_frame_quality), '-pix_fmt', 'rgb24', '-vf', 'fps=' + str(fps), os.path.join(temp_directory_path, '%04d.' + 'png')])

def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1

execution_threads = suggest_execution_threads()

def create_queue(temp_frame_paths: List[str]) -> Queue[str]:
    queue: Queue[str] = Queue()
    for frame_path in temp_frame_paths:
        queue.put(frame_path)
    return queue

def pick_queue(queue: Queue[str], queue_per_future: int) -> List[str]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues

def update_progress(progress: Any = None) -> None:
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
    progress.set_postfix({
        'memory_usage': '{:.2f}'.format(memory_usage).zfill(5) + 'GB',
        'execution_providers': execution_providers,
        'execution_threads': execution_threads
    })
    progress.refresh()
    progress.update(1)

def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], update: Callable[[], None]) -> None:
    with ThreadPoolExecutor(max_workers=execution_threads) as executor:
        futures = []
        queue = create_queue(temp_frame_paths)
        queue_per_future = max(len(temp_frame_paths) // execution_threads, 1)
        while not queue.empty():
            future = executor.submit(process_frames, source_path, pick_queue(queue, queue_per_future), update)
            futures.append(future)
        for future in as_completed(futures):
            future.result()

def process_video(source_path: str, frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        multi_process_frame(source_path, frame_paths, process_frames, lambda: update_progress(progress))

def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]

def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]

def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())

execution_providers = decode_execution_providers(suggest_execution_providers())

def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = face_swapper_path
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=execution_providers)
    return FACE_SWAPPER

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

similar_face_distance = 0.85

def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                if distance < similar_face_distance:
                    return face
    return None

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    target_face = find_similar_face(temp_frame, reference_face)
    if target_face:
        temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    reference_frame = cv2.imread(temp_frame_paths[0])
    reference_face = get_one_face(reference_frame)
    total_frames = total = len(temp_frame_paths)
    progress_bar = st.progress(0)
    start_time = time.time()
    time_text = st.text("Time Remaining: ")
    for i, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        elapsed_time = time.time() - start_time
        frames_completed = i+1
        frames_remaining = total_frames - frames_completed
        time_remaining = (frames_remaining / frames_completed) * elapsed_time
        progress_bar.progress(frames_completed / total_frames) 
        if update:
            update()
    time_text.empty()
    progress_bar.empty()

# def get_face_enhancer() -> Any:
#     global FACE_ENHANCER

#     with THREAD_LOCK:
#         if FACE_ENHANCER is None:
#             model_path = '/content/drive/MyDrive/FaceSwap3/GFPGANv1.4.pth'
#             # todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
#             FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_device())
#     return FACE_ENHANCER

def get_device() -> str:
    if 'CUDAExecutionProvider' in execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in execution_providers:
        return 'mps'
    return 'cpu'

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    padding_x = int((end_x - start_x) * 0.5)
    padding_y = int((end_y - start_y) * 0.5)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size:
        with THREAD_SEMAPHORE:
            _, _, temp_face = get_face_enhancer().enhance(temp_face, paste_back=True)
        temp_frame[start_y:end_y, start_x:end_x] = temp_face
    return temp_frame

def enhance_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame

def enhance_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = enhance_frame(None, None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()

def create_video(target_path: str, temp_directory_path: str, fps: float = 30) -> bool:
    temp_output_path = os.path.join(temp_directory_path, 'temp.mp4')
    output_video_quality = (35 + 1) * 51 // 100
    commands = ['-hwaccel', 'auto', '-r', str(fps), '-i', os.path.join(temp_directory_path, '%04d.' + 'png'), '-c:v', output_video_encoder]
    if output_video_encoder in ['libx264', 'libx265', 'libvpx']:
        commands.extend(['-crf', str(output_video_quality)])
    if output_video_encoder in ['h264_nvenc', 'hevc_nvenc']:
        commands.extend(['-cq', str(output_video_quality)])
    commands.extend(['-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', temp_output_path])
    return run_ffmpeg(commands)

def move_temp(temp_output_path: str, output_path: str) -> None:
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)

def restore_audio(target_path: str, temp_directory_path: str, output_path: str) -> None:
    temp_output_path = os.path.join(temp_directory_path, 'temp.mp4')
    done = run_ffmpeg(['-i', temp_output_path, '-i', target_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y', output_path])
    if not done:
        move_temp(temp_output_path, output_path)

def start() -> Optional[str]:
    if not pre_start():
        return
    with st.spinner("Preparing..."):
        target_directory_path = os.path.dirname(target_path)
        temp_directory_path = os.path.join(target_directory_path, 'temp')
        Path(temp_directory_path).mkdir(parents=True, exist_ok=True)
        fps = detect_fps(target_path)
    with st.spinner(f'Extracting frames with {fps} FPS...'):
        extract_frames(target_path, temp_directory_path, fps)
        temp_frame_paths = glob.glob((os.path.join(glob.escape(temp_directory_path), '*.' + 'png')))
    if temp_frame_paths:
        with st.spinner('Swapping Progressing...'):
            process_video(source_path, temp_frame_paths, process_frames)
        # with st.spinner('Enhancing Progressing...')
            # process_video(None, temp_frame_paths, enhance_frames)
    else:
        st.write('Frames not found...')
        return
    with st.spinner(f'Creating video with {fps} FPS...'):
        create_video(target_path, temp_directory_path, fps)
    with st.spinner('Restoring audio...'):
        restore_audio(target_path, temp_directory_path, output_path)
    if is_video(output_path):
        st.video(cv2.imread(output_path))
    else:
        st.write('Processing to video failed!')

def conditional_download(download_directory_path: str, urls: List[str]) -> Optional[str]:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get('Content-Length', 0))
            with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))  # type: ignore[attr-defined]

def run() -> Optional[str]:
    if not pre_check():
        return
    conditional_download(face_swapper_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
    conditional_download(face_enhancer_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'])
    start()
