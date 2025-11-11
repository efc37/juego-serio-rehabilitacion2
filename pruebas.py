import cv2
import numpy as np
import random
import time
import mediapipe as mp
from config import config  # tu config con model_path

# ========= MediaPipe Tasks (como el profe) =========
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=config.model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)

# ========= Parámetros del juego =========
SHOW_DURATION = 5.0
FRUIT_COLORS = [
    (0, 0, 255),
    (0, 255, 255),
    (0, 128, 0),
    (255, 0, 255),
    (0, 165, 255),
    (255, 0, 0),
]

sequence_generated = False
current_sequence = []
start_show_time = None

waiting_for_user = False
current_step = 0
TAP_COOLDOWN = 0.5
last_tap = 0

correct_hits = []          # guardamos los ítems acertados
disabled_indices = set()   # círculos que ya no reaccionan

last_error = None
last_error_time = 0

# ===================================================
def draw_fruit_count(frame, center, r, color, count):
    cx, cy = center
    mini_r = max(8, r // 6)
    offset = int(r * 0.5)

    # 9 posiciones máximo (3x3)
    positions = []
    for row in [-1, 0, 1]:
        for col in [-1, 0, 1]:
            px = cx + int(col * offset / 1.5)
            py = cy + int(row * offset / 1.5)
            positions.append((px, py))

    for i in range(count):
        if i >= len(positions):
            break
        x, y = positions[i]
        cv2.circle(frame, (x, y), mini_r, color, -1)
        cv2.circle(frame, (x, y), mini_r, (0, 0, 0), 1)

def draw_cross(frame, center, r):
    x, y = center
    cv2.line(frame, (x - r//3, y - r//3), (x + r//3, y + r//3), (0, 0, 255), 4)
    cv2.line(frame, (x - r//3, y + r//3), (x + r//3, y - r//3), (0, 0, 255), 4)

# ===================================================
# Abrimos cámara dentro del contexto del landmarker
with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # necesitamos un timestamp creciente para detect_for_video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    frame_ms = int(1000 / fps)
    timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # convertir a mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # detectar pose
        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += frame_ms

        # ====== cuadrícula 3x4 ======
        r = int(min(w / 16, h / 9))
        x_margin = int(w * 0.18)
        y_margin = int(h * 0.20)
        x_positions = np.linspace(x_margin, w - x_margin, 4, dtype=int)
        y_positions = np.linspace(y_margin, h - y_margin, 3, dtype=int)
        centers = [(x, y) for y in y_positions for x in x_positions]  # 12 círculos

        # generar secuencia SOLO una vez
        if not sequence_generated:
            chosen_indices = random.sample(range(len(centers)), 6)
            seq = []
            for i, idx in enumerate(chosen_indices):
                color = FRUIT_COLORS[i % len(FRUIT_COLORS)]
                count = i + 1
                seq.append({"idx": idx, "color": color, "count": count})
            current_sequence = seq
            start_show_time = time.time()
            sequence_generated = True

        # dibujar líneas (parando en el borde)
        for y in y_positions:
            for i in range(len(x_positions) - 1):
                x1 = x_positions[i] + r
                x2 = x_positions[i + 1] - r
                cv2.line(frame, (x1, y), (x2, y), (200, 200, 200), 2)

        for x in x_positions:
            for j in range(len(y_positions) - 1):
                y1 = y_positions[j] + r
                y2 = y_positions[j + 1] - r
                cv2.line(frame, (x, y1), (x, y2), (200, 200, 200), 2)

        # dibujar círculos base
        for (cx, cy) in centers:
            cv2.circle(frame, (cx, cy), r, (0, 0, 0), 2)

        now = time.time()

        # ===== fase de mostrar =====
        if now - start_show_time <= SHOW_DURATION:
            for item in current_sequence:
                idx = item["idx"]
                color = item["color"]
                count = item["count"]
                center = centers[idx]
                draw_fruit_count(frame, center, r, color, count)
        else:
            waiting_for_user = True

        # ===== fase de interacción (con landmarker) =====
        if waiting_for_user and result.pose_landmarks:
            # solo tomamos la primera persona
            person_landmarks = result.pose_landmarks[0]
            # puedes cambiar RIGHT_INDEX por RIGHT_WRIST si te da más estabilidad
            idx_lm = mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value
            lm = person_landmarks[idx_lm]
            hand_x = int(lm.x * w)
            hand_y = int(lm.y * h)

            # dibujar el punto de la mano
            cv2.circle(frame, (hand_x, hand_y), 8, (255, 0, 0), -1)

            if now - last_tap > TAP_COOLDOWN:
                for idx_circle, (cx, cy) in enumerate(centers):
                    # si ya está acertado, lo ignoramos
                    if idx_circle in disabled_indices:
                        continue

                    dx = hand_x - cx
                    dy = hand_y - cy
                    if dx*dx + dy*dy <= r*r:
                        # círculo tocado
                        expected_circle = current_sequence[current_step]["idx"]
                        if idx_circle == expected_circle:
                            # acierto → guardamos ese ítem y bloqueamos
                            hit_item = current_sequence[current_step]
                            correct_hits.append(hit_item)
                            disabled_indices.add(idx_circle)
                            current_step += 1
                            if current_step >= len(current_sequence):
                                waiting_for_user = False
                        else:
                            # fallo → cruz 2 s
                            last_error = (cx, cy)
                            last_error_time = now
                        last_tap = now
                        break

        # dibujar aciertos permanentes (bolitas correctas)
        for hit in correct_hits:
            idx = hit["idx"]
            color = hit["color"]
            count = hit["count"]
            center = centers[idx]
            draw_fruit_count(frame, center, r, color, count)

        # dibujar cruz temporal (2 s)
        if last_error is not None and now - last_error_time < 2.0:
            draw_cross(frame, last_error, r)

        cv2.imshow("Juego Frutal", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



'''import cv2
import numpy as np
import random
import time
import mediapipe as mp

# Iniciar cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SHOW_DURATION = 5.0

FRUIT_COLORS = [
    (0, 0, 255),
    (0, 255, 255),
    (0, 128, 0),
    (255, 0, 255),
    (0, 165, 255),
    (255, 0, 0),
]

sequence_generated = False
current_sequence = []
start_show_time = None

waiting_for_user = False
current_step = 0
TAP_COOLDOWN = 0.5
last_tap = 0

correct_hits = []          # lista de ítems acertados
disabled_indices = set()   # índices de círculos que ya no se pueden volver a tocar

last_error = None
last_error_time = 0


def draw_fruit_count(frame, center, r, color, count):
    cx, cy = center
    mini_r = max(8, r // 6)
    offset = int(r * 0.5)

    positions = []
    for row in [-1, 0, 1]:
        for col in [-1, 0, 1]:
            px = cx + int(col * offset / 1.5)
            py = cy + int(row * offset / 1.5)
            positions.append((px, py))

    for i in range(count):
        if i >= len(positions):
            break
        x, y = positions[i]
        cv2.circle(frame, (x, y), mini_r, color, -1)
        cv2.circle(frame, (x, y), mini_r, (0, 0, 0), 1)


def draw_cross(frame, center, r):
    x, y = center
    cv2.line(frame, (x - r//3, y - r//3), (x + r//3, y + r//3), (0, 0, 255), 4)
    cv2.line(frame, (x - r//3, y + r//3), (x + r//3, y - r//3), (0, 0, 255), 4)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    r = int(min(w / 16, h / 9))
    x_margin = int(w * 0.18)
    y_margin = int(h * 0.20)
    x_positions = np.linspace(x_margin, w - x_margin, 4, dtype=int)
    y_positions = np.linspace(y_margin, h - y_margin, 3, dtype=int)
    centers = [(x, y) for y in y_positions for x in x_positions]

    if not sequence_generated:
        chosen_indices = random.sample(range(len(centers)), 6)
        seq = []
        for i, idx in enumerate(chosen_indices):
            color = FRUIT_COLORS[i % len(FRUIT_COLORS)]
            count = i + 1
            seq.append({"idx": idx, "color": color, "count": count})
        current_sequence = seq
        start_show_time = time.time()
        sequence_generated = True

    # Dibujar líneas
    for y in y_positions:
        for i in range(len(x_positions) - 1):
            x1 = x_positions[i] + r
            x2 = x_positions[i + 1] - r
            cv2.line(frame, (x1, y), (x2, y), (200, 200, 200), 2)

    for x in x_positions:
        for j in range(len(y_positions) - 1):
            y1 = y_positions[j] + r
            y2 = y_positions[j + 1] - r
            cv2.line(frame, (x, y1), (x, y2), (200, 200, 200), 2)

    # Dibujar círculos base
    for (cx, cy) in centers:
        cv2.circle(frame, (cx, cy), r, (0, 0, 0), 2)

    now = time.time()

    # Fase de mostrar frutas
    if now - start_show_time <= SHOW_DURATION:
        for item in current_sequence:
            idx = item["idx"]
            color = item["color"]
            count = item["count"]
            center = centers[idx]
            draw_fruit_count(frame, center, r, color, count)
    else:
        waiting_for_user = True

    # Fase de interacción
    if waiting_for_user and results.pose_landmarks:
        lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        hand_x = int(lm.x * w)
        hand_y = int(lm.y * h)
        cv2.circle(frame, (hand_x, hand_y), 8, (255, 0, 0), -1)

        if now - last_tap > TAP_COOLDOWN:
            for idx_circle, (cx, cy) in enumerate(centers):
                # si este círculo ya fue acertado → ignorar
                if idx_circle in disabled_indices:
                    continue

                dx = hand_x - cx
                dy = hand_y - cy
                if dx * dx + dy * dy <= r * r:
                    expected_circle = current_sequence[current_step]["idx"]
                    if idx_circle == expected_circle:
                        #ACERTÓ 
                        hit_item = current_sequence[current_step]
                        correct_hits.append(hit_item)
                        disabled_indices.add(idx_circle)
                        current_step += 1
                        if current_step >= len(current_sequence):
                            waiting_for_user = False
                    else:
                        #FALLO 
                        last_error = (cx, cy)
                        last_error_time = now
                    last_tap = now
                    break

    # Dibujar aciertos permanentes (bolitas)
    for hit in correct_hits:
        idx = hit["idx"]
        color = hit["color"]
        count = hit["count"]
        center = centers[idx]
        draw_fruit_count(frame, center, r, color, count)

    # Dibujar cruz de fallo temporal (2s)
    if last_error is not None and now - last_error_time < 2.0:
        draw_cross(frame, last_error, r)

    cv2.imshow("Juego Frutal", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pose.close()'''
