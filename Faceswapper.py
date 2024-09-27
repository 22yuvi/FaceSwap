import cv2
import insightface
import numpy as np
import onnxruntime
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import tqdm as tqdm
import os
import urllib
import threading
from typing import List, Optional, Any, Callable
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(working_dir,"source.png")
target_path = os.path.join(working_dir,"target.mp4")
source_name, _ = os.path.splitext(os.path.basename(source_path))
target_name, target_extension = os.path.splitext(os.path.basename(target_path))
output_path = os.path.join(working_dir, source_name + '-' + target_name + target_extension)
face_swapper_path = working_dir
face_enhancer_path = working_dir

# Set up ONNX Runtime to use CUDA
onnxruntime.set_default_logger_severity(3)
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Global variables
FACE_ANALYSER = None
FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()

def get_face_analyser():
    global FACE_ANALYSER
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER

def get_face_swapper():
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = os.path.join('/content/FaceSwap/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=providers)
    return FACE_SWAPPER

def get_many_faces(frame):
    face_analyser = get_face_analyser()
    return face_analyser.get(frame)

def swap_face(source_face, target_face, frame):
    face_swapper = get_face_swapper()
    return face_swapper.get(frame, target_face, source_face, paste_back=True)

def process_frame(source_face, frame):
    faces = get_many_faces(frame)
    if faces:
        for face in faces:
            frame = swap_face(source_face, face, frame)
    return frame

def process_video(source_path, target_path, output_path):
    source_face = get_many_faces(cv2.imread(source_path))[0]
    
    video = cv2.VideoCapture(target_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for _ in range(total_frames):
            ret, frame = video.read()
            if not ret:
                break
            future = executor.submit(process_frame, source_face, frame)
            futures.append(future)
        
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing frames"):
            frame = future.result()
            output_video.write(frame)
    
    video.release()
    output_video.release()

# Main execution
def start(quality: bool) -> Optional[str]:    
    process_video(source_path, target_path, output_path)

def conditional_download(download_directory_path: str, urls: List[str]) -> Optional[str]:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get('Content-Length', 0))
            with tqdm.tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))  # type: ignore[attr-defined]

def run(quality: bool) -> Optional[str]:
    with st.spinner("Preprocessing..."):
        conditional_download(face_swapper_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
        conditional_download(face_enhancer_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'])
    start(quality)
