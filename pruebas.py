import cv2
import numpy as np
import random
import time
import sys
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
    num_hands=2
)

# ========= Parámetros del juego =========
FRUIT_COLORS = [
    (180, 105, 255),   # rosa fuerte
    (220, 170, 255),   # rosa pastel visible
    (60, 0, 139),      # granate
    (180, 50, 130),    # morado
    (255, 220, 150),   # azul claro
    (180, 255, 200),   # verde clarito
    (255, 255, 180),   # amarillo clarito
]

sequence_generated = False
current_sequence = []
start_show_time = None

waiting_for_user = False
current_step = 0

correct_hits = []          # ítems acertados (para redibujarlos)
disabled_indices = set()   # círculos que ya no reaccionan

last_error = None
last_error_time = 0

# ========= Puntuaciones =========
score_time = 0               # puntos por rapidez
score_combo = 0              # bonus de combos (combo_len * 2 por cada racha >= 2)
combo = 0                    # racha actual de aciertos seguidos
best_combo = 0               # mejor racha alcanzada
hits = 0                     # número de aciertos
fails = 0                    # número de fallos
sequence_start_time = None   # para cronómetro global (interno)
last_hit_time = None         # para medir rapidez entre aciertos
game_over = False

# HUD de último evento (se muestra solo tras acierto/fallo)
last_ui_event_time = 0.0
UI_SHOW_SECONDS = 2.0
last_ui_score = 0
last_ui_combo = 0

# ========= Pantalla de inicio =========
show_start_screen = True     # hasta que el usuario pulse ENTER

# Conexiones para dibujar el "esqueleto" de la mano
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),        # índice
    (0, 9), (9, 10), (10, 11), (11, 12),   # medio
    (0, 13), (13, 14), (14, 15), (15, 16), # anular
    (0, 17), (17, 18), (18, 19), (19, 20)  # meñique
]


def count_extended_fingers(hand_landmarks):
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    extended = 0

    for tip_idx, pip_idx in finger_pairs:
        tip = hand_landmarks[tip_idx]
        pip = hand_landmarks[pip_idx]
        if tip.y < pip.y:
            extended += 1

    return extended


def compute_time_points(dt):
    """
    Devuelve puntos por rapidez según el tiempo (dt en segundos)
    que has tardado en encontrar el siguiente círculo correcto.
    """
    if dt <= 3.0:
        return 10
    elif dt <= 5.0:
        return 5
    elif dt <= 8.0:
        return 2
    else:
        return 0


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
        now = time.time()

        # ========= PANTALLA DE INICIO =========
        if show_start_screen:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (200, 180, 255), -1)
            alpha = 0.45
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            title = "MotionMind"
            subtitle = "Pulsa ENTER para empezar"
            hint = "Pulsa ESC para salir"

            cx_mid = w // 2
            cy_mid = h // 2

            def put_centered(text, y, scale, color, thickness=2):
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                cv2.putText(frame, text, (cx_mid - tw // 2, y), font, scale, color, thickness)

            put_centered(title,    cy_mid - 40, 1.7, (255, 255, 255), 3)
            put_centered(subtitle, cy_mid + 10, 0.9, (80, 80, 80), 2)
            put_centered(hint,     cy_mid + 60, 0.7, (80, 80, 80), 2)

            cv2.imshow("Juego serio", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 13:  # ENTER
                show_start_screen = False

            continue  # saltamos la lógica del juego en este frame

        # ========= LÓGICA DEL JUEGO =========

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
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
            start_show_time = now
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

        # ===== fase de mostrar secuencia =====
        if not game_over:
            if now - start_show_time <= config.frute_time:
                for item in current_sequence:
                    idx = item["idx"]
                    color = item["color"]
                    count = item["count"]
                    center = centers[idx]
                    draw_fruit_count(frame, center, r, color, count)
            else:
                if not waiting_for_user:
                    waiting_for_user = True
                    sequence_start_time = now
                    last_hit_time = now

        # ===== fase de interacción =====
        if not game_over and waiting_for_user and result.hand_landmarks:
            selection_done = False

            for hand_landmarks in result.hand_landmarks:
                extended = count_extended_fingers(hand_landmarks)
                is_open_palm = (extended >= 4)

                lm_index = hand_landmarks[8]
                hand_x = int(lm_index.x * w)
                hand_y = int(lm_index.y * h)

                # dibujar landmarks
                for lm in hand_landmarks:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    cv2.circle(frame, (px, py), 5, (203, 192, 255), -1)

                for start_idx, end_idx in HAND_CONNECTIONS:
                    p1 = hand_landmarks[start_idx]
                    p2 = hand_landmarks[end_idx]
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (220, 208, 255), 2)

                if is_open_palm:
                    for idx_circle, (cx, cy) in enumerate(centers):
                        if idx_circle in disabled_indices:
                            continue

                        dx = hand_x - cx
                        dy = hand_y - cy
                        if dx*dx + dy*dy <= r*r:
                            expected_circle = current_sequence[current_step]["idx"]

                            if idx_circle == expected_circle:
                                # ========= ✅ ACIERTO =========
                                hit_item = current_sequence[current_step]
                                correct_hits.append(hit_item)
                                disabled_indices.add(idx_circle)
                                current_step += 1

                                # contar acierto
                                hits += 1

                                # tiempo desde el último acierto
                                if last_hit_time is None:
                                    dt = now - sequence_start_time
                                else:
                                    dt = now - last_hit_time
                                last_hit_time = now

                                time_points = compute_time_points(dt)
                                score_time += time_points

                                # combo actual
                                combo += 1
                                if combo > best_combo:
                                    best_combo = combo

                                # si completamos todos los círculos → finalizamos combo también
                                if current_step >= len(current_sequence):
                                    waiting_for_user = False
                                    game_over = True
                                    if combo >= 2:
                                        score_combo += combo * 2
                                        combo = 0

                                # actualizar HUD
                                score_base = max(0, hits * 10 - fails * 5)
                                total_score = score_base + score_time + score_combo
                                last_ui_event_time = now
                                last_ui_score = total_score
                                last_ui_combo = combo

                            else:
                                # ========= ❌ FALLO =========
                                # Solo contamos el fallo si NO estamos ya dibujando una X reciente
                                if (last_error is None) or (now - last_error_time >= 2.0):
                                    last_error = (cx, cy)
                                    last_error_time = now

                                    # contar fallo
                                    fails += 1

                                    # cerrar combo si había racha >= 2
                                    if combo >= 2:
                                        score_combo += combo * 2
                                    combo = 0

                                    # actualizar HUD
                                    score_base = max(0, hits * 10 - fails * 5)
                                    total_score = score_base + score_time + score_combo
                                    last_ui_event_time = now
                                    last_ui_score = total_score
                                    last_ui_combo = combo
                                # si aún seguimos dentro de los 2s de la X anterior, no sumamos más fallos

                            selection_done = True
                            break

                if selection_done or game_over:
                    break

        # dibujar aciertos permanentes
        for hit in correct_hits:
            idx = hit["idx"]
            color = hit["color"]
            count = hit["count"]
            center = centers[idx]
            draw_fruit_count(frame, center, r, color, count)

        # dibujar cruz temporal (X) durante 2 segundos
        if last_error is not None and now - last_error_time < 2.0:
            draw_cross(frame, last_error, r)

        # ===== HUD de puntuación tras acierto/fallo =====
        if not game_over and (now - last_ui_event_time) < UI_SHOW_SECONDS:
            y0 = 80
            cv2.putText(
                frame,
                f"Score: {last_ui_score}",
                (30, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                2
            )
            cv2.putText(
                frame,
                f"Combo: {last_ui_combo}",
                (30, y0 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )

        # ===== Pantalla final =====
        if game_over:
            # por si acaso termina sin fallo y aún queda combo abierto
            if combo >= 2:
                score_combo += combo * 2
                combo = 0

            score_base = max(0, hits * 10 - fails * 5)
            total_score = score_base + score_time + score_combo

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            title = "Juego terminado"
            line1 = f"Score base + bonus: {score_base+score_combo}  (hits={hits}, fails={fails})"
            line3 = f"Puntos por rapidez: {score_time}"
            line4 = f"Total: {total_score}"
            line5 = f"Mejor combo: {best_combo}"
            hint  = "Pulsa ESC para salir"

            font = cv2.FONT_HERSHEY_SIMPLEX
            cx_mid = w // 2
            cy_mid = h // 2

            def put_centered(text, y, scale, color, thickness=2):
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                cv2.putText(frame, text, (cx_mid - tw // 2, y), font, scale, color, thickness)

            put_centered(title,  cy_mid - 100, 1.5, (255, 255, 255), 3)
            put_centered(line1,  cy_mid - 50, 0.8, (255, 220, 220), 2)
            put_centered(line3,  cy_mid + 20, 0.8, (255, 220, 220), 2)
            put_centered(line4,  cy_mid + 55, 0.9, (255, 255, 255), 2)
            put_centered(line5,  cy_mid + 90, 0.8, (255, 255, 200), 2)
            put_centered(hint,   cy_mid + 130, 0.8, (200, 200, 200), 2)

        cv2.imshow("Juego serio", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
