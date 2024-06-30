import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import tempfile
from tqdm import tqdm
import time
import cv2
import moviepy.editor as mp
from PIL import Image
from streamlit_image_select import image_select
import glob
from FinalFaceSwapCode import run

# Set page configuration
st.set_page_config(page_title="Face Swapper",
                   layout="wide",
                   page_icon="🧑‍⚕️")

with st.sidebar:
    selected = option_menu('Options',
                           ['Use Available Images',
                            'Upload Custom Images'],
                           menu_icon='list',
                           icons=['person', 'upload'],
                           default_index=0)
# getting the working directory of the main.py


st.title('Face Swap using Inswapper')
st.write("""
## Face Swapper
##### Upload a video and replace the face in the video with available faces or upload a custom image for face swap.
""")

working_dir = os.path.dirname(os.path.abspath(__file__))
male_images = os.path.join(working_dir, "male")
female_images = os.path.join(working_dir, "female")

if selected == 'Use Available Images':
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        gender = option_menu('Select Gender',
                              ['Male',
                                'Female'],
                              menu_icon='person',
                              icons=['gender-male', 'gender-female'],
                              default_index=0)  
    with col2:
      quality = option_menu('Select Output Quality',
                              ['High',
                                'Normal'],
                              menu_icon='person',
                              default_index=0)
    if gender == 'Male':
        imgs = glob.glob((os.path.join(glob.escape(male_images), '*.' + 'jpg')))
        image = image_select(
            label="Male",
            images=imgs,
            captions=[os.path.splitext(os.path.basename(image))[0] for image in imgs],
        )
    if gender == 'Female':
        imgs = glob.glob((os.path.join(glob.escape(male_images), '*.' + 'jpg')))
        image = image_select(
            label="Female",
            images=imgs,
            captions=[os.path.splitext(os.path.basename(image))[0] for image in imgs],
        )
    img = Image.open(image)

if selected == 'Upload Custom Images':
    uploaded_file = st.file_uploader("Upload your images here...", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension in ['.jpg', '.png', '.jpeg']:
            img = Image.open(uploaded_file)
            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                st.image(img)

uploaded_file = st.file_uploader("Upload your video here...", type=['mp4', 'mov', 'avi', 'mkv'])

if st.button("Swap"):
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            # Save the video file to a temporary location
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())

            audio = mp.AudioFileClip(temp_file.name)
            video = cv2.VideoCapture(temp_file.name)

            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                st.markdown('<p style="text-align: center;">Before</p>', unsafe_allow_html=True)
                st.video(temp_file.name)

            with col2:
                st.markdown('<p style="text-align: center;">After</p>', unsafe_allow_html=True)

                with st.spinner("Swapping Faces..."):
                    output_frames = []
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress_bar = st.progress(0)  # Create a progress bar

                    start_time = time.time()
                    time_text = st.text("Time Remaining: ")  # Initialize text value
                    fps = video.get(cv2.CAP_PROP_FPS)
                    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Codec for .mp4 files
                    output_path = os.path.join(working_dir, "target.mp4")
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    for _ in tqdm(range(total_frames), unit='frame', desc="Progress"):
                        ret, frame = video.read()
                        out.write(frame) 
                        if not ret:
                            break
                    source_img = os.path.join(working_dir, "source.png")
                    img.save(source_img)
                    out.release()
                    converted_filename = run()
                    st.video(cv2.imread(converted_filename))
                  
                st.download_button(
                            label="Download Colorized Video",
                            data=open(converted_filename.name, "rb").read(),
                            file_name="swapped_video.mp4"
                        )


    
