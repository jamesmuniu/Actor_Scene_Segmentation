import streamlit as st
import os
import cv2
import tempfile
import pickle
from moviepy.editor import VideoFileClip, concatenate_videoclips
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from deepface import DeepFace
import matplotlib.pyplot as plt
import shutil

# Set page config
st.set_page_config(page_title="🎮 Actor Scene Segmentor", layout="wide")
st.title("🎮 Actor Scene Segmentor")
st.markdown("""
Upload a video and images of known actors. This app segments scenes, identifies where each actor appears, summarizes screen time, and visualizes results.
""")

# Sidebar - Upload
st.sidebar.header("Upload Inputs")
video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
actor_images = st.sidebar.file_uploader("Upload actor images (with recognizable names)", accept_multiple_files=True, type=["jpg", "png"])

# Utility functions
def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    return scene_list

def identify_actors(video_path, scene_list, db_path):
    actor_times = {}
    cap = cv2.VideoCapture(video_path)

    for start_time, end_time in scene_list:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time.get_seconds() * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            result = DeepFace.find(frame, db_path=db_path, enforce_detection=False)
            for r in result:
                for _, row in r.iterrows():
                    identity = row['identity'].split('/')[-1].split('.')[0]
                    actor_times[identity] = actor_times.get(identity, 0) + (end_time.get_seconds() - start_time.get_seconds())
        except Exception as e:
            st.warning(f"Face detection failed in scene: {e}")

    cap.release()
    return actor_times

def plot_actor_screen_time(actor_times, db_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(actor_times))
    bars = ax.barh(y_pos, list(actor_times.values()), color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(actor_times.keys())
    ax.set_xlabel('Screen Time (s)')
    ax.set_title('Actor Screen Time Summary')

    # Add images to bars
    for i, actor in enumerate(actor_times.keys()):
        actor_img_path_jpg = os.path.join(db_path, actor + ".jpg")
        actor_img_path_png = os.path.join(db_path, actor + ".png")
        actor_img_path = actor_img_path_jpg if os.path.exists(actor_img_path_jpg) else actor_img_path_png
        if os.path.exists(actor_img_path):
            img = cv2.imread(actor_img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (40, 40))
            ax.imshow(img, extent=[-5, 0, i - 0.4, i + 0.4], aspect='auto')

    st.pyplot(fig)

# Main App Execution
if video_file and actor_images:
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        db_path = os.path.join(temp_dir, "actors")
        os.makedirs(db_path, exist_ok=True)

        for image in actor_images:
            with open(os.path.join(db_path, image.name), "wb") as f:
                f.write(image.read())

        st.success("Processing video and actor database...")
        scene_list = detect_scenes(video_path)
        st.write(f"Detected {len(scene_list)} scenes")

        actor_times = identify_actors(video_path, scene_list, db_path)
        st.subheader("✅ Actor Appearance Summary")
        st.write(actor_times)

        # Save results as a pickled model
        result_model = {
            "scene_list": scene_list,
            "actor_times": actor_times
        }
        model_path = os.path.join(temp_dir, "actor_scene_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(result_model, f)

        st.download_button("📁 Download Scene Model (.pkl)", data=open(model_path, "rb"), file_name="actor_scene_model.pkl")

        st.subheader("📊 Actor Screen Time Plot")
        plot_actor_screen_time(actor_times, db_path)

else:
    st.info("Upload a video and at least one actor image to begin.")
