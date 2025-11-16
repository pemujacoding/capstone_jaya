import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

# --- Inisialisasi MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Fungsi Hitung Gaze Ratio (Versi Akurat & Stabil) ---
def get_gaze_ratio(eye_points, facial_landmarks, frame_width, frame_height, is_left_eye=True):
    try:
        # Titik mata
        eye_left_point = (
            int(facial_landmarks[eye_points[0]].x * frame_width),
            int(facial_landmarks[eye_points[0]].y * frame_height)
        )
        eye_right_point = (
            int(facial_landmarks[eye_points[3]].x * frame_width),
            int(facial_landmarks[eye_points[3]].y * frame_height)
        )
        eye_top_point = (
            int(facial_landmarks[eye_points[1]].x * frame_width),
            int(facial_landmarks[eye_points[1]].y * frame_height)
        )
        eye_bottom_point = (
            int(facial_landmarks[eye_points[2]].x * frame_width),
            int(facial_landmarks[eye_points[2]].y * frame_height)
        )

        # Gunakan 4 landmark iris (lebih stabil)
        if is_left_eye:
            iris_indices = [468, 469, 470, 471]  # kiri
        else:
            iris_indices = [473, 474, 475, 476]  # kanan

        iris_x = np.mean([facial_landmarks[i].x * frame_width for i in iris_indices])
        iris_y = np.mean([facial_landmarks[i].y * frame_height for i in iris_indices])

        eye_width = abs(eye_right_point[0] - eye_left_point[0])
        eye_height = abs(eye_bottom_point[1] - eye_top_point[1])

        if eye_width == 0 or eye_height == 0:
            return "N/A", 0.5, 0.5

        horizontal_ratio = (iris_x - eye_left_point[0]) / (eye_width + 1e-6)
        vertical_ratio = (iris_y - eye_top_point[1]) / (eye_height + 1e-6)

        # Batasi ke 0–1
        horizontal_ratio = np.clip(horizontal_ratio, 0, 1)
        vertical_ratio = np.clip(vertical_ratio, 0, 1)

        return "Fokus", horizontal_ratio, vertical_ratio

    except:
        return "N/A", 0.5, 0.5


# --- Streamlit UI ---
st.set_page_config(page_title="Analisis Gaze Mata", layout="wide")
st.title("👁️ Analisis Gaze Mata untuk Deteksi 'Cheating'")
st.write("Upload video untuk menganalisis arah pandangan dan mendeteksi potensi *cheating*.")

uploaded_file = st.file_uploader("📂 Pilih file video...", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("🚀 Mulai Analisis"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Tidak bisa membuka file video.")
        else:
            st.info("📸 Kalibrasi: Tatap layar dengan tenang selama 3 detik...")
            time.sleep(1)

            baseline_h, baseline_v = [], []
            start_time = time.time()

            # --- Kalibrasi baseline (3 detik) ---
            while time.time() - start_time < 3:
                ret, frame = cap.read()
                if not ret:
                    break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        left_eye = [33, 160, 158, 133]
                        right_eye = [362, 385, 387, 263]
                        _, lh, lv = get_gaze_ratio(left_eye, face_landmarks.landmark, frame.shape[1], frame.shape[0], True)
                        _, rh, rv = get_gaze_ratio(right_eye, face_landmarks.landmark, frame.shape[1], frame.shape[0], False)
                        baseline_h.append((lh + rh) / 2)
                        baseline_v.append((lv + rv) / 2)

            base_h = np.median(baseline_h) if len(baseline_h) > 0 else 0.5
            base_v = np.median(baseline_v) if len(baseline_v) > 0 else 0.5

            st.success("✅ Kalibrasi selesai! Memulai analisis video...")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            st_frame = st.empty()
            st_progress = st.progress(0)
            st_summary_chart = st.empty()
            st_summary_text = st.empty()

            gaze_data = {
                "Fokus": 0,
                "Melirik Kiri": 0,
                "Melirik Kanan": 0,
                "Melirik Atas": 0,
                "Melirik Bawah": 0,
                "N/A": 0
            }

            # --- Parameter stabilisasi ---
            tolerance_h = 0.12
            tolerance_v = 0.20
            alpha = 0.25  # smoothing factor

            prev_h, prev_v = base_h, base_v
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)
                frame_output = frame.copy()
                status = "N/A"

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        left_eye = [33, 160, 158, 133]
                        right_eye = [362, 385, 387, 263]
                        _, lh, lv = get_gaze_ratio(left_eye, face_landmarks.landmark, frame.shape[1], frame.shape[0], True)
                        _, rh, rv = get_gaze_ratio(right_eye, face_landmarks.landmark, frame.shape[1], frame.shape[0], False)
                        h_ratio = (lh + rh) / 2
                        v_ratio = (lv + rv) / 2

                        # Smoothing (rolling average)
                        h_ratio = alpha * h_ratio + (1 - alpha) * prev_h
                        v_ratio = alpha * v_ratio + (1 - alpha) * prev_v
                        prev_h, prev_v = h_ratio, v_ratio

                        # --- Deteksi arah ---
                        if abs(h_ratio - base_h) < tolerance_h and abs(v_ratio - base_v) < tolerance_v:
                            status = "Fokus"
                        elif h_ratio - base_h > tolerance_h:
                            status = "Melirik Kiri"
                        elif base_h - h_ratio > tolerance_h:
                            status = "Melirik Kanan"
                        elif v_ratio - base_v > tolerance_v:
                            status = "Melirik Bawah"
                        elif base_v - v_ratio > tolerance_v:
                            status = "Melirik Atas"

                        cv2.putText(frame_output, status, (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                gaze_data[status] += 1
                st_frame.image(frame_output, channels="BGR", use_container_width=True)

                if total_frames > 0:
                    st_progress.progress(frame_count / total_frames)

            cap.release()
            st_progress.progress(1.0)
            st.success("✅ Analisis Selesai!")

            # --- Ringkasan Hasil ---
            if frame_count > 0:
                chart_data = {
                    "Arah Pandangan": list(gaze_data.keys()),
                    "Jumlah Frame": list(gaze_data.values())
                }
                st_summary_chart.bar_chart(chart_data, x="Arah Pandangan", y="Jumlah Frame")

                total_valid = frame_count - gaze_data["N/A"]
                fokus_percentage = (gaze_data["Fokus"] / total_valid) * 100 if total_valid > 0 else 0
                melirik_total = sum([
                    gaze_data["Melirik Kiri"],
                    gaze_data["Melirik Kanan"],
                    gaze_data["Melirik Atas"],
                    gaze_data["Melirik Bawah"]
                ])
                melirik_percentage = (melirik_total / total_valid) * 100 if total_valid > 0 else 0

                st_summary_text.subheader(f"📊 Ringkasan Analisis (Total {frame_count} frame):")
                st_summary_text.write(f"**Persentase Fokus:** `{fokus_percentage:.2f}%`")
                st_summary_text.write(f"**Frekuensi Melirik:** `{melirik_percentage:.2f}%`")

                # --- Analisis Cheating ---
                CHEATING_THRESHOLD = 40.0
                MELIRIK_THRESHOLD = 30.0

                if fokus_percentage < CHEATING_THRESHOLD or melirik_percentage > MELIRIK_THRESHOLD:
                    st.error(f"⚠️ Potensi *Cheating*: Fokus rendah ({fokus_percentage:.1f}%) atau sering melirik ({melirik_percentage:.1f}%).")
                elif fokus_percentage < 60:
                    st.warning(f"🟡 Kemungkinan Tidak Fokus: Fokus hanya {fokus_percentage:.1f}%.")
                else:
                    st.success("🟢 Subjek tampak fokus dan tidak menunjukkan tanda-tanda cheating.")
