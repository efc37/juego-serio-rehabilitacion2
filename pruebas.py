

import cv2
import numpy as np
import random
import time
import mediapipe as mp
from config import config 

# ========= MediaPipe Tasks =========
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=config.model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1  # solo una mano
)

# ========= Parámetros del juego =========
SHOW_DURATION = 5.0
DWELL_TIME = 2.0      # segundos que hay que mantener la mano en el círculo
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

correct_hits = []          # guardamos los ítems acertados (para redibujarlos)
disabled_indices = set()   # círculos que ya no reaccionan

last_error = None
last_error_time = 0

# NUEVO: para controlar cuánto tiempo llevamos sobre un círculo
hover_circle_idx = None
hover_start_time = 0.0


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


with HandLandmarker.create_from_options(hand_options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

        # pasar frame a formato mediapipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
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

        # ===== fase de interacción (con HAND) =====
        if waiting_for_user and result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            lm = hand_landmarks[8]  # punta índice
            hand_x = int(lm.x * w)
            hand_y = int(lm.y * h)

            # dibujar punto mano
            cv2.circle(frame, (hand_x, hand_y), 8, (255, 0, 0), -1)

            # detectar en qué círculo está ahora la mano
            current_circle = None
            for idx_circle, (cx, cy) in enumerate(centers):
                if idx_circle in disabled_indices:
                    continue  # este ya no reacciona
                dx = hand_x - cx
                dy = hand_y - cy
                if dx*dx + dy*dy <= r*r:
                    current_circle = idx_circle
                    break

            if current_circle is not None:
                # si es el mismo círculo que el frame anterior, contamos tiempo
                if current_circle == hover_circle_idx:
                    # cuánto llevamos aquí
                    dwell = now - hover_start_time
                    # dibujar un pequeño círculo de progreso opcional
                    # cv2.circle(frame, centers[current_circle], r//3, (0,255,0), 2)
                    if dwell >= DWELL_TIME:
                        # aquí confirmamos la selección
                        expected_circle = current_sequence[current_step]["idx"]
                        cx, cy = centers[current_circle]
                        if current_circle == expected_circle:
                            # acierto
                            hit_item = current_sequence[current_step]
                            correct_hits.append(hit_item)
                            disabled_indices.add(current_circle)
                            current_step += 1
                            if current_step >= len(current_sequence):
                                waiting_for_user = False
                        else:
                            # fallo
                            last_error = (cx, cy)
                            last_error_time = now
                        # después de confirmar, reseteamos el hover
                        hover_circle_idx = None
                        hover_start_time = 0.0
                else:
                    # hemos entrado en un círculo nuevo → empezar a contar
                    hover_circle_idx = current_circle
                    hover_start_time = now
            else:
                # no estamos sobre ningún círculo → reset
                hover_circle_idx = None
                hover_start_time = 0.0

        # dibujar aciertos permanentes
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




'''
import cv2
import numpy as np
import random
import time
import mediapipe as mp
from config import config 

# ========= MediaPipe Tasks =========
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=config.model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1  # solo una mano
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

correct_hits = []          # guardamos los ítems acertados (para redibujarlos)
disabled_indices = set()   # círculos que ya no reaccionan

last_error = None
last_error_time = 0


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


# ========= Bucle principal con HandLandmarker =========
with HandLandmarker.create_from_options(hand_options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

        # pasar frame a formato mediapipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
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

        # ===== fase de interacción (con HAND) =====
        if waiting_for_user and result.hand_landmarks:
            # primera mano detectada
            hand_landmarks = result.hand_landmarks[0]
            # 8 = tip del índice
            lm = hand_landmarks[8]
            hand_x = int(lm.x * w)
            hand_y = int(lm.y * h)

            # dibujar el punto de la mano para depurar
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
'''