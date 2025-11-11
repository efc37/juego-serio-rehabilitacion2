import cv2
import numpy as np
import random
import time
import mediapipe as mp

# Iniciar cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Mediapipe Pose (como en la práctica)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Duración en segundos que se muestran las "frutas" (puntos de colores)
SHOW_DURATION = 5.0

# Colores que simulan frutas (BGR)
FRUIT_COLORS = [
    (0, 0, 255),    # rojo
    (0, 255, 255),  # amarillo
    (0, 128, 0),    # verde
    (255, 0, 255),  # magenta
    (0, 165, 255),  # naranja
    (255, 0, 0),    # azul
]

# Variables de control
sequence_generated = False
current_sequence = []   # lista de dicts: {"idx": ..., "color": ..., "count": ...}
start_show_time = None

# Para la fase de "el usuario repite"
waiting_for_user = False
current_step = 0        # qué fruta/círculo toca ahora
last_feedback_time = 0
last_feedback = None    # ("ok", circle_center) o ("fail", circle_center)
TAP_COOLDOWN = 0.5      # segundos para no contar varios toques seguidos
last_tap = 0

def draw_fruit_count(frame, center, r, color, count):
    """Dibuja 'count' bolitas de color dentro del círculo grande."""
    cx, cy = center
    mini_r = max(8, r // 6)
    offset = int(r * 0.5)

    # Posiciones tipo 3x3 dentro del círculo
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

def draw_tick(frame, center, r):
    x, y = center
    # dibujar un tick verde dentro del círculo
    # lo hacemos con dos líneas
    cv2.line(frame, (x - r//3, y), (x - r//10, y + r//3), (0, 255, 0), 4)
    cv2.line(frame, (x - r//10, y + r//3), (x + r//3, y - r//4), (0, 255, 0), 4)

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

    # pasamos a RGB para mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # Parámetros de cuadrícula
    r = int(min(w / 16, h / 9))
    x_margin = int(w * 0.18)
    y_margin = int(h * 0.20)
    x_positions = np.linspace(x_margin, w - x_margin, 4, dtype=int)
    y_positions = np.linspace(y_margin, h - y_margin, 3, dtype=int)
    centers = [(x, y) for y in y_positions for x in x_positions]  # 12 centros

    # Generar la secuencia solo una vez
    if not sequence_generated:
        chosen_indices = random.sample(range(len(centers)), 6)
        seq = []
        for i, idx in enumerate(chosen_indices):
            color = FRUIT_COLORS[i % len(FRUIT_COLORS)]
            count = i + 1   # 1,2,3,4,5,6
            seq.append({"idx": idx, "color": color, "count": count})
        current_sequence = seq
        start_show_time = time.time()
        sequence_generated = True

    # Dibujar líneas (hasta los bordes de los círculos)
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

    # Mostrar las "frutas" (puntos) durante SHOW_DURATION segundos
    if now - start_show_time <= SHOW_DURATION:
        for item in current_sequence:
            idx = item["idx"]
            color = item["color"]
            count = item["count"]
            center = centers[idx]
            draw_fruit_count(frame, center, r, color, count)
    else:
        # aquí empieza la fase de "el jugador repite"
        waiting_for_user = True

    # Si estamos esperando al usuario, miramos la mano
    if waiting_for_user and results.pose_landmarks:
        # tomamos el dedo índice derecho
        lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        hand_x = int(lm.x * w)
        hand_y = int(lm.y * h)

        # dibujar punto de la mano para depurar
        cv2.circle(frame, (hand_x, hand_y), 8, (255, 0, 0), -1)

        # comprobamos si ha pasado suficiente tiempo desde el último toque
        if now - last_tap > TAP_COOLDOWN:
            # comprobamos si está dentro de algún círculo
            for idx_circle, (cx, cy) in enumerate(centers):
                dx = hand_x - cx
                dy = hand_y - cy
                if dx*dx + dy*dy <= r*r:
                    # ha tocado un círculo
                    expected_circle = current_sequence[current_step]["idx"]
                    if idx_circle == expected_circle:
                        # acierto
                        last_feedback = ("ok", (cx, cy))
                        last_feedback_time = now
                        current_step += 1
                        # si ha terminado toda la secuencia, podrías reiniciar o mostrar mensaje
                        if current_step >= len(current_sequence):
                            waiting_for_user = False  # ya acabó
                    else:
                        # fallo
                        last_feedback = ("fail", (cx, cy))
                        last_feedback_time = now
                    last_tap = now
                    break  # ya no seguimos mirando otros círculos

    # dibujar feedback (tick o cruz) durante un momento
    if last_feedback is not None and now - last_feedback_time < 1.0:  # 1 segundo de feedback
        kind, center = last_feedback
        if kind == "ok":
            draw_tick(frame, center, r)
        else:
            draw_cross(frame, center, r)

    cv2.imshow("Juego Frutal", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
