import cv2
import numpy as np
import random
import time
import sys
import os
import csv
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

USERS_FILE = "usuarios.csv"
DATA_FILE = "datos_juego_serio.csv"

# Conexiones para dibujar el "esqueleto" de la mano
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),        # índice
    (0, 9), (9, 10), (10, 11), (11, 12),   # medio
    (0, 13), (13, 14), (14, 15), (15, 16), # anular
    (0, 17), (17, 18), (18, 19), (19, 20)  # meñique
]


# ========= FUNCIONES AUXILIARES =========

def count_extended_fingers(hand_landmarks):
    # Índices (tip, pip) para cada dedo (sin pulgar):
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
    if dt <= 1.0:
        return 10
    elif dt <= 2.0:
        return 5
    elif dt <= 3.0:
        return 2
    else:
        return 0


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


def put_centered(frame, text, y, scale, color, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    cx_mid = w // 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(frame, text, (cx_mid - tw // 2, y), font, scale, color, thickness)


def ensure_users_file():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["username", "password"])


def register_user(cap):
    ensure_users_file()
    username = ""
    password = ""
    error_msg = ""
    entering_password = False  # False → escribiendo usuario, True → escribiendo contraseña

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Fondo rosita translúcido
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (200, 180, 255), -1)
        alpha = 0.45
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        title = "Registro - MotionMind"
        label_user = "Nombre de usuario (no puede repetirse):"
        label_pass = "Contraseña:"
        hint = "ENTER para continuar / cambiar campo | ESC para volver"

        show_user = f"> {username}_"
        show_pass = "> " + ("*" * len(password) + "_")

        put_centered(frame, title,      120, 1.2, (255, 255, 255), 3)
        put_centered(frame, label_user, 190, 0.8, (60, 60, 60), 2)
        put_centered(frame, show_user,  230, 0.9, (30, 30, 30), 2)
        put_centered(frame, label_pass, 290, 0.8, (60, 60, 60), 2)
        put_centered(frame, show_pass,  330, 0.9, (30, 30, 30), 2)
        put_centered(frame, hint,       400, 0.7, (80, 80, 80), 2)

        if error_msg:
            put_centered(frame, error_msg, 450, 0.7, (0, 0, 255), 2)

        cv2.imshow("Juego serio", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC → volver al menú
            return None, None

        elif key in (13, 10):  # ENTER
            if not entering_password:
                # Pasamos de usuario a contraseña
                if not username.strip():
                    error_msg = "El nombre no puede estar vacío."
                else:
                    error_msg = ""
                    entering_password = True
            else:
                # Intentamos registrar
                if not password:
                    error_msg = "La contraseña no puede estar vacía."
                else:
                    # Comprobamos que no exista el usuario
                    exists = False
                    with open(USERS_FILE, newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row["username"] == username:
                                exists = True
                                break
                    if exists:
                        error_msg = "Ese nombre de usuario ya existe. Elige otro."
                    else:
                        # Guardamos usuario nuevo
                        with open(USERS_FILE, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([username, password])
                        return username, password

        elif key in (8, 127):  # Backspace
            if not entering_password:
                username = username[:-1]
            else:
                password = password[:-1]

        elif 32 <= key <= 126:  # caracteres imprimibles
            ch = chr(key)
            if not entering_password:
                if len(username) < 20:
                    username += ch
            else:
                if len(password) < 20:
                    password += ch


def login_user(cap):
    ensure_users_file()
    username = ""
    password = ""
    error_msg = ""
    entering_password = False

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (200, 180, 255), -1)
        alpha = 0.45
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        title = "Inicio de sesión - MotionMind"
        label_user = "Nombre de usuario:"
        label_pass = "Contraseña:"
        hint = "ENTER para continuar / cambiar campo | ESC para volver"

        show_user = f"> {username}_"
        show_pass = "> " + ("*" * len(password) + "_")

        put_centered(frame, title,      120, 1.2, (255, 255, 255), 3)
        put_centered(frame, label_user, 190, 0.8, (60, 60, 60), 2)
        put_centered(frame, show_user,  230, 0.9, (30, 30, 30), 2)
        put_centered(frame, label_pass, 290, 0.8, (60, 60, 60), 2)
        put_centered(frame, show_pass,  330, 0.9, (30, 30, 30), 2)
        put_centered(frame, hint,       400, 0.7, (80, 80, 80), 2)

        if error_msg:
            put_centered(frame, error_msg, 450, 0.7, (0, 0, 255), 2)

        cv2.imshow("Juego serio", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC → volver al menú
            return None, None

        elif key in (13, 10):  # ENTER
            if not entering_password:
                if not username.strip():
                    error_msg = "El nombre no puede estar vacío."
                else:
                    error_msg = ""
                    entering_password = True
            else:
                if not password:
                    error_msg = "La contraseña no puede estar vacía."
                else:
                    # Comprobamos credenciales
                    ok = False
                    with open(USERS_FILE, newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row["username"] == username and row["password"] == password:
                                ok = True
                                break
                    if ok:
                        return username, password
                    else:
                        error_msg = "Usuario o contraseña incorrectos."

        elif key in (8, 127):  # Backspace
            if not entering_password:
                username = username[:-1]
            else:
                password = password[:-1]

        elif 32 <= key <= 126:  # caracteres imprimibles
            ch = chr(key)
            if not entering_password:
                if len(username) < 20:
                    username += ch
            else:
                if len(password) < 20:
                    password += ch


def auth_menu(cap):
    """Devuelve (username, password) tras registro/login correcto, o sale."""
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (200, 180, 255), -1)
        alpha = 0.45
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        title = "MotionMind"
        option1 = "[1] Registro"
        option2 = "[2] Inicio de sesión"
        hint = "Pulsa 1 o 2 para elegir | ESC para salir"

        put_centered(frame, title,   180, 1.7, (255, 255, 255), 3)
        put_centered(frame, option1, 250, 1.0, (50, 50, 50), 2)
        put_centered(frame, option2, 300, 1.0, (50, 50, 50), 2)
        put_centered(frame, hint,    370, 0.8, (80, 80, 80), 2)

        cv2.imshow("Juego serio", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord('1'):
            u, p = register_user(cap)
            if u is not None:
                return u, p
        elif key == ord('2'):
            u, p = login_user(cap)
            if u is not None:
                return u, p


# ========= PROGRAMA PRINCIPAL =========

with HandLandmarker.create_from_options(hand_options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    frame_ms = int(1000 / fps)
    timestamp = 0

    # --- Pantalla de menú: registro / login ---
    username, _ = auth_menu(cap)

    # ========= INICIALIZAR ESTADO DEL JUEGO =========
    sequence_generated = False
    current_sequence = []
    start_show_time = None

    waiting_for_user = False
    current_step = 0

    correct_hits = []          # ítems acertados (para redibujarlos)
    disabled_indices = set()   # círculos que ya no reaccionan

    last_error = None
    last_error_time = 0

    # Puntuaciones
    score_base = 0               # puntos base + combo
    score_time = 0               # puntos por rapidez
    combo = 0                    # racha de aciertos seguidos
    best_combo = 0               # mejor racha alcanzada
    sequence_start_time = None   # para cronómetro global (interno)
    last_hit_time = None         # para medir rapidez entre aciertos
    game_over = False

    # HUD
    last_ui_event_time = 0.0
    UI_SHOW_SECONDS = 2.0
    last_ui_score = 0
    last_ui_combo = 0

    # ========= BUCLE PRINCIPAL DEL JUEGO =========
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

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
                    sequence_start_time = now  # arranca cronómetro interno
                    last_hit_time = now       # para rapidez del primer círculo

        # ===== fase de interacción =====
        if not game_over and waiting_for_user and result.hand_landmarks:
            selection_done = False  # para cortar si alguna mano selecciona

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

                # esqueleto
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
                                # ✅ acierto
                                hit_item = current_sequence[current_step]
                                correct_hits.append(hit_item)
                                disabled_indices.add(idx_circle)
                                current_step += 1

                                # rapidez
                                if last_hit_time is None:
                                    dt = now - sequence_start_time
                                else:
                                    dt = now - last_hit_time
                                last_hit_time = now

                                time_points = compute_time_points(dt)

                                # puntos base + combo
                                base_points = 10
                                combo += 1
                                if combo > best_combo:
                                    best_combo = combo
                                combo_points = combo * 2

                                score_base += base_points + combo_points
                                score_time += time_points
                                total_score = score_base + score_time

                                last_ui_event_time = now
                                last_ui_score = total_score
                                last_ui_combo = combo

                                if current_step >= len(current_sequence):
                                    waiting_for_user = False
                                    game_over = True

                            else:
                                # ❌ fallo
                                last_error = (cx, cy)
                                last_error_time = now

                                score_base -= 5
                                if score_base < 0:
                                    score_base = 0
                                combo = 0

                                total_score = score_base + score_time
                                last_ui_event_time = now
                                last_ui_score = total_score
                                last_ui_combo = combo

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

        # dibujar cruz temporal
        if last_error is not None and now - last_error_time < 2.0:
            draw_cross(frame, last_error, r)

        # HUD
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

        # ===== Pantalla final con rehabilitación =====
        if game_over:
            total_score = score_base + score_time

            # Guardar en CSV de datos de juego
            file_exists = os.path.exists(DATA_FILE)
            with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["nombre", "total_score", "score_time", "best_combo", "timestamp"])
                writer.writerow([
                    username,
                    total_score,
                    score_time,
                    best_combo,
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ])

            # Leer historial del usuario
            user_scores = []
            try:
                with open(DATA_FILE, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row["nombre"] == username:
                            user_scores.append(float(row["total_score"]))
            except FileNotFoundError:
                pass

            sesiones = len(user_scores)
            if sesiones >= 2:
                prev_score = user_scores[-2]
                if total_score > prev_score:
                    evo_text = "Has mejorado respecto a tu última sesión. ¡Buen trabajo!"
                elif total_score < prev_score:
                    evo_text = "Hoy ha bajado un poco, pero es parte del proceso. ¡Sigue así!"
                else:
                    evo_text = "Has mantenido la misma puntuación que en la última sesión. Estabilidad 👍"
            else:
                evo_text = "Esta es tu primera sesión registrada. A partir de aquí veremos tu evolución."

            best_hist_score = max(user_scores) if user_scores else total_score

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            title = f"Juego terminado - {username}"
            line1 = f"Score base (incluye combo): {score_base}"
            line2 = f"Puntos por rapidez: {score_time}"
            line3 = f"Total partida: {total_score}"
            line4 = f"Mejor combo en esta sesión: {best_combo}"
            line5 = f"Sesiones registradas: {sesiones} | Mejor total histórico: {best_hist_score}"
            line6 = evo_text
            hint  = "Pulsa ESC para salir"

            put_centered(frame, title,  160, 1.3, (255, 255, 255), 3)
            put_centered(frame, line1,  210, 0.8, (255, 220, 220), 2)
            put_centered(frame, line2,  245, 0.8, (255, 220, 220), 2)
            put_centered(frame, line3,  280, 0.9, (255, 255, 255), 2)
            put_centered(frame, line4,  315, 0.8, (255, 255, 200), 2)
            put_centered(frame, line5,  350, 0.7, (220, 220, 220), 2)
            put_centered(frame, line6,  385, 0.7, (220, 220, 255), 2)
            put_centered(frame, hint,   430, 0.8, (200, 200, 200), 2)

        cv2.imshow("Juego serio", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
