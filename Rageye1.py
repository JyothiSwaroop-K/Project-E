


import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from streamlit_mic_recorder import mic_recorder
import whisper
import io
import os
from gtts import gTTS
import base64
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import cv2
import mediapipe as mp
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
from datetime import datetime

st.set_page_config(page_title="Python Voice Quiz", layout="centered")
st.title("🎤 Python Voice Quiz (Strict Evaluation)")

# ---------------------- SESSION STATE INIT ---------------------
st.session_state.setdefault('selected_level', None)
st.session_state.setdefault('user_info_collected', False)
st.session_state.setdefault('user_name', "")
st.session_state.setdefault('questions', [])
st.session_state.setdefault('current_q', 0)
st.session_state.setdefault('answers', [])
st.session_state.setdefault('submitted', False)
st.session_state.setdefault('evaluations', [])
st.session_state.setdefault('eye_tracking_started', False)
st.session_state.setdefault('eye_tracking_stopped', False)
st.session_state.setdefault('question_metadatas', [])
st.session_state.setdefault('audio_played', False)
st.session_state.setdefault('results_audio_generated', False)

USER_DATA_FILE = "user_data.csv"
EYE_TRACKING_FILE = "eye_tracking_data.csv"

# ---------------------- DATA SAVE FUNCTIONS ---------------------
def save_user_data(name, email, phone):
    file_exists = os.path.isfile(USER_DATA_FILE)
    with open(USER_DATA_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Name", "Email", "Phone"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, email, phone])

def save_eye_tracking_data(user_name, eye_data):
    file_exists = os.path.isfile(EYE_TRACKING_FILE)
    with open(EYE_TRACKING_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["User_Name", "Timestamp", "Eye_Aspect_Ratio", "Looking_At_Screen", "Test_Timestamp"])
        test_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i, timestamp in enumerate(eye_data['timestamp']):
            writer.writerow([
                user_name,
                datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f"),
                eye_data['ear'][i],
                eye_data['looking_at_screen'][i],
                test_timestamp
            ])

# ---------------------- EYE TRACKING ---------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
eye_tracking_active = False
eye_tracking_data = defaultdict(list)
eye_tracking_thread = None
cap = None
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_points, landmarks):
    A = np.linalg.norm(np.array([landmarks[eye_points[1]].x - landmarks[eye_points[5]].x,
                                  landmarks[eye_points[1]].y - landmarks[eye_points[5]].y]))
    B = np.linalg.norm(np.array([landmarks[eye_points[2]].x - landmarks[eye_points[4]].x,
                                  landmarks[eye_points[2]].y - landmarks[eye_points[4]].y]))
    C = np.linalg.norm(np.array([landmarks[eye_points[0]].x - landmarks[eye_points[3]].x,
                                  landmarks[eye_points[0]].y - landmarks[eye_points[3]].y]))
    return (A + B) / (2.0 * C)

def eye_tracking():
    global cap
    cap = cv2.VideoCapture(0)
    while eye_tracking_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
                right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
                ear = (left_ear + right_ear) / 2.0
                timestamp = time.time()
                eye_tracking_data['timestamp'].append(timestamp)
                eye_tracking_data['ear'].append(ear)
                eye_tracking_data['looking_at_screen'].append(1 if ear > 0.2 else 0)
        time.sleep(0.1)
    if cap:
        cap.release()

def start_eye_tracking():
    global eye_tracking_active, eye_tracking_thread
    eye_tracking_active = True
    eye_tracking_data.clear()
    eye_tracking_thread = threading.Thread(target=eye_tracking, daemon=True)
    eye_tracking_thread.start()
    st.session_state.eye_tracking_started = True

def stop_eye_tracking():
    global eye_tracking_active
    eye_tracking_active = False
    if eye_tracking_thread and eye_tracking_thread.is_alive():
        eye_tracking_thread.join(timeout=2)
    if cap:
        cap.release()
    st.session_state.eye_tracking_stopped = True

def plot_eye_tracking_results():
    if not eye_tracking_data or not eye_tracking_data.get('timestamp'):
        return None
    df = pd.DataFrame(eye_tracking_data)
    df['time_elapsed'] = df['timestamp'] - df['timestamp'].min()
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['green' if x == 1 else 'red' for x in df['looking_at_screen']]
    ax.scatter(df['time_elapsed'], df['looking_at_screen'], c=colors, alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Attention (1 = Looking)')
    ax.set_yticks([0, 1])
    ax.grid(True)
    return fig

# ---------------------- AUDIO ---------------------
def autoplay_audio(audio_bytes):
    audio_str = "data:audio/wav;base64,%s" % (base64.b64encode(audio_bytes).decode())
    audio_html = f"""
    <audio autoplay>
        <source src="{audio_str}" type="audio/wav">
    </audio>
    """
    st.components.v1.html(audio_html, height=0)

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file.read()

# ---------------------- TABS ---------------------
def safe_show_tabs(evaluations, questions, answers):
    if not evaluations:
        st.warning("No evaluations available. Please ensure you answered all questions.")
        return
    tabs = st.tabs([f"Q{i+1}" for i in range(len(evaluations))])
    for i, tab in enumerate(tabs):
        with tab:
            eval_data = evaluations[i]
            st.markdown(f"**Question {i+1}:** {questions[i]}")
            st.code(answers[i])
            st.markdown(f"**Score:** {eval_data['score']}/5")
            st.progress(eval_data['score'] / 5)
            st.markdown("**Details:**")
            st.write(eval_data['justification'])
            st.markdown("**Reference Answer:**")
            st.info(eval_data['reference_answer'])

# ---------------------- EMBEDDINGS & EVALUATION ---------------------
@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path="python_quiz_db")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(name="python_questions", embedding_function=embedding_function)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    langchain_chroma = Chroma(client=client, collection_name="python_questions", embedding_function=embeddings)
    return collection, langchain_chroma

chroma_collection, langchain_chroma = get_chroma_collection()
model = SentenceTransformer("all-MiniLM-L6-v2")

def keyword_match(user_answer, reference_answer):
    ref_words = set(re.findall(r'\b\w+\b', reference_answer.lower()))
    user_words = set(re.findall(r'\b\w+\b', user_answer.lower()))
    return len(ref_words & user_words)

def strict_evaluation(question, user_answer):
    if not user_answer.strip():
        return {"score": 0, "justification": "Empty answer provided", "reference_answer": ""}
    try:
        result = chroma_collection.query(query_texts=[question], n_results=1)
        ref_answer = result['documents'][0][0]
    except:
        ref_answer = "Reference answer not found."
    emb_user = model.encode([user_answer])[0].reshape(1, -1)
    emb_ref = model.encode([ref_answer])[0].reshape(1, -1)
    similarity = cosine_similarity(emb_user, emb_ref)[0][0] * 100
    keywords_matched = keyword_match(user_answer, ref_answer)
    if similarity >= 50 and keywords_matched >= 6:
        score = 5
    elif 40 <= similarity and keywords_matched >= 4:
        score = 4
    elif 30 <= similarity and keywords_matched >= 3:
        score = 3
    elif 25 <= similarity and keywords_matched >= 2:
        score = 2
    elif 15 <= similarity and keywords_matched >= 1:
        score = 1
    else:
        score = 0
    return {
        "score": score,
        "justification": f"Similarity: {similarity:.2f}%\nKeywords matched: {keywords_matched}",
        "reference_answer": ref_answer
    }

# ✅ Now integrated safely into your main code. You can now call `safe_show_tabs(...)` without crashing.


# ------------------------- STREAMLIT STATE INIT ---------------------
if 'selected_level' not in st.session_state:
    st.session_state.selected_level = None

if 'user_info_collected' not in st.session_state:
    st.session_state.user_info_collected = False

if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

if 'questions' not in st.session_state:
    st.session_state.update({
        'questions': [],
        'current_q': 0,
        'answers': [],
        'submitted': False,
        'question_metadatas': [],
        'last_voice_result': None,
        'audio_played': False,
        'evaluations': [],
        'results_audio_generated': False,
        'eye_tracking_started': False,
        'eye_tracking_stopped': False
    })

# ------------------------- USER INFO COLLECTION ---------------------
if not st.session_state.user_info_collected:
    with st.form("user_info_form"):
        st.subheader("Please Enter Your Details")
        name = st.text_input("Full Name*", placeholder="Enter your full name")
        email = st.text_input("Email*", placeholder="Enter your email")
        phone = st.text_input("Phone Number*", placeholder="Enter your phone number")
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            if name and email and phone:
                # Save user data
                save_user_data(name, email, phone)
                st.session_state.user_info_collected = True
                st.session_state.user_name = name
                st.rerun()
            else:
                st.error("Please fill in all required fields (marked with *)")

# ------------------------- DIFFICULTY SELECTION ---------------------
if st.session_state.user_info_collected and st.session_state.selected_level is None:
    st.subheader("Select Difficulty Level:")
    col1, col2, col3 = st.columns(3)
    if col1.button("🟢 Beginner"):
        st.session_state.selected_level = "Beginner"
    if col2.button("🟠 Intermediate"):
        st.session_state.selected_level = "Intermediate"
    if col3.button("🔴 Advanced"):
        st.session_state.selected_level = "Advanced"

if st.session_state.selected_level:
    level_colors = {"Beginner": "🟢", "Intermediate": "🟠", "Advanced": "🔴"}
    st.markdown(f"### Selected Difficulty: {level_colors[st.session_state.selected_level]} {st.session_state.selected_level}")

# ------------------------- GET QUESTIONS ---------------------
def get_questions_by_difficulty(level, n=5):
    try:
        results = chroma_collection.query(
            query_texts=[f"Python {level} level questions"],
            n_results=20,
            where={"difficulty": level}
        )
        import random
        combined = list(zip(results['documents'][0], results['metadatas'][0]))
        random.shuffle(combined)
        selected = combined[:n]
        questions = [q[0] for q in selected]
        metadatas = [q[1] for q in selected]
        return questions, metadatas
    except Exception as e:
        st.error(f"Error fetching questions: {str(e)}")
        return [], []

# ------------------------- QUIZ ---------------------
if st.session_state.selected_level and not st.session_state.questions:
    if st.button("🎤 Start Voice Quiz"):
        questions, metadatas = get_questions_by_difficulty(st.session_state.selected_level, 5)
        if questions:
            st.session_state.questions = questions
            st.session_state.question_metadatas = metadatas
            st.session_state.current_q = 0
            st.session_state.answers = [''] * len(questions)
            st.session_state.submitted = False
            st.session_state.evaluations = []
            st.session_state.last_voice_result = None
            st.session_state.audio_played = False
            st.session_state.results_audio_generated = False
            st.session_state.eye_tracking_stopped = False
            
            # Start eye tracking when quiz starts
            start_eye_tracking()
            st.session_state.eye_tracking_started = True
            st.success("Eye tracking started! The system is now monitoring your attention during the quiz.")
            st.rerun()

if st.session_state.questions and not st.session_state.submitted:
    q_index = st.session_state.current_q

    if not st.session_state.audio_played:
        question_text = f"Question {q_index + 1}. {st.session_state.questions[q_index]}"
        audio_bytes = text_to_speech(question_text)
        autoplay_audio(audio_bytes)
        st.session_state.audio_played = True

    st.subheader(f"Question {q_index + 1} of {len(st.session_state.questions)}")
    difficulty = st.session_state.question_metadatas[q_index]['difficulty']
    badge = "🟢 Beginner" if difficulty == "Beginner" else "🟠 Intermediate" if difficulty == "Intermediate" else "🔴 Advanced"
    st.markdown(badge)
    st.markdown("**Speak Your Answer:**")

    # Show eye tracking status
    if st.session_state.eye_tracking_started and not st.session_state.eye_tracking_stopped:
        st.info("👁️ Eye tracking is active - Please look at the screen during the quiz")

    if st.session_state.answers[q_index]:
        st.markdown("**Your Current Answer:**")
        st.success(st.session_state.answers[q_index])

    audio_data = mic_recorder(
        start_prompt="🎤 Start Recording Answer",
        stop_prompt="⏹ Stop Recording",
        just_once=True,
        key=f'recording_{q_index}',
        use_container_width=True
    )

    if audio_data:
        temp_audio_path = f"temp_input_{q_index}.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_data['bytes'])
        if os.path.exists(temp_audio_path):
            model = whisper.load_model("base")
            result = model.transcribe(temp_audio_path)
            transcribed_text = result['text']
            st.session_state.last_voice_result = transcribed_text
            st.session_state.answers[q_index] = transcribed_text
            os.remove(temp_audio_path)
            st.rerun()

    col1, col2 = st.columns(2)
    if q_index > 0 and col2.button("Previous Question"):
        st.session_state.current_q -= 1
        st.session_state.audio_played = False
        st.rerun()

    if col1.button("Next Question"):
        if q_index < len(st.session_state.questions) - 1:
            st.session_state.current_q += 1
            st.session_state.audio_played = False
            st.rerun()

    if st.button("🔊 Replay Question"):
        question_text = f"Question {q_index + 1}. {st.session_state.questions[q_index]}"
        audio_bytes = text_to_speech(question_text)
        autoplay_audio(audio_bytes)

    if q_index == len(st.session_state.questions) - 1:
        if st.button("✅ Submit Test"):
            with st.spinner("Stopping eye tracking and evaluating answers..."):
                # Stop eye tracking immediately when test is submitted
                if st.session_state.eye_tracking_started and not st.session_state.eye_tracking_stopped:
                    stop_eye_tracking()
                    st.session_state.eye_tracking_stopped = True
                    
                    # Save eye tracking data to CSV
                    if eye_tracking_data and st.session_state.user_name:
                        save_eye_tracking_data(st.session_state.user_name, eye_tracking_data)
                
                # Evaluate answers
                evaluations = []
                progress_bar = st.progress(0)
                for i, (q, a) in enumerate(zip(st.session_state.questions, st.session_state.answers)):
                    evaluations.append(strict_evaluation(q, a))
                    progress_bar.progress((i + 1) / len(st.session_state.questions))
                st.session_state.evaluations = evaluations
                st.session_state.submitted = True
                st.rerun()

# ------------------------- RESULTS ---------------------
if st.session_state.submitted:
    st.header("📊 Detailed Evaluation Report")
    total_score = sum(eval['score'] for eval in st.session_state.evaluations)
    max_score = len(st.session_state.questions) * 5

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Score", f"{total_score}/{max_score}")
        col2.metric("Percentage", f"{(total_score/max_score)*100:.1f}%")
        col3.metric("Performance Level",
                    "Excellent" if total_score/max_score >= 0.8 else
                    "Good" if total_score/max_score >= 0.6 else
                    "Needs Practice")

    # Display eye tracking results - ONLY in results section
    st.subheader("👀 Eye Tracking Analysis")
    
    if st.session_state.eye_tracking_stopped:
        eye_fig = plot_eye_tracking_results()
        if eye_fig:
            st.pyplot(eye_fig)
            
            # Calculate attention metrics
            df = pd.DataFrame(eye_tracking_data)
            if len(df) > 0:
                total_time = df['timestamp'].max() - df['timestamp'].min() if len(df) > 1 else 0
                attention_samples = sum(df['looking_at_screen'])
                total_samples = len(df)
                attention_percentage = (attention_samples / total_samples) * 100 if total_samples > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Test Duration", f"{total_time:.1f} seconds")
                col2.metric("Attention Percentage", f"{attention_percentage:.1f}%")
                col3.metric("Data Points Captured", f"{total_samples}")
                
                # Additional insights
                if attention_percentage >= 80:
                    st.success("🟢 Excellent attention! You maintained focus throughout the test.")
                elif attention_percentage >= 60:
                    st.warning("🟡 Good attention, but there's room for improvement.")
                else:
                    st.error("🔴 Poor attention detected. Please ensure you're looking at the screen during tests.")
                
                st.caption("The graph shows binary attention data: Green dots (1) = Looking at screen, Red dots (0) = Not looking at screen")
        else:
            st.warning("⚠️ No eye tracking data was collected during the test. Please ensure your camera is working properly.")
    else:
        st.info("Processing eye tracking data...")

    # Question-wise results
    tab_list = st.tabs([f"Q{i+1}" for i in range(len(st.session_state.evaluations))])

    for i, tab in enumerate(tab_list):
        with tab:
            eval_data = st.session_state.evaluations[i]
            with st.expander("Question & Answer", expanded=True):
                st.markdown(f"**Question {i+1}:** {st.session_state.questions[i]}")
                st.code(st.session_state.answers[i], language="python")

            with st.expander("Evaluation Details", expanded=True):
                st.markdown(f"**Score:** {eval_data['score']}/5")
                st.progress(eval_data['score']/5)
                st.markdown("**Details:**")
                st.write(eval_data['justification'])
                st.markdown("**Reference Answer:**")
                st.info(eval_data['reference_answer'])

    if st.button("🔁 Retake Quiz"):
        # Reset all session state
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.evaluations = []
        st.session_state.submitted = False
        st.session_state.current_q = 0
        st.session_state.audio_played = False
        st.session_state.results_audio_generated = False
        st.session_state.eye_tracking_started = False
        st.session_state.eye_tracking_stopped = False
        
        # Clear eye tracking data for new quiz
        eye_tracking_data.clear()
        
        # Ensure eye tracking is fully stopped
        stop_eye_tracking()
        st.rerun()

    if not st.session_state.get('results_audio_generated'):
        with st.spinner("Generating audio summary..."):
            summary = f"Your scored {total_score} out of {max_score}. "
            summary += "Excellent work!" if total_score/max_score >= 0.8 else \
                       "Good job!" if total_score/max_score >= 0.6 else \
                       "Keep practicing!"
            audio_bytes = text_to_speech(summary)
            autoplay_audio(audio_bytes)
            st.session_state.results_audio_generated = True


