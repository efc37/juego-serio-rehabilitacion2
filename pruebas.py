import cv2
import numpy as np
import random
import time

# Iniciar cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Duración en segundos que se muestran las frutas
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
current_sequence = []
start_show_time = None

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Parámetros de cuadrícula
    r = int(min(w / 16, h / 9))
    x_margin = int(w * 0.18)
    y_margin = int(h * 0.20)
    x_positions = np.linspace(x_margin, w - x_margin, 4, dtype=int)
    y_positions = np.linspace(y_margin, h - y_margin, 3, dtype=int)
    centers = [(x, y) for y in y_positions for x in x_positions]

    # Generar la secuencia solo una vez
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

    # Mostrar frutas durante 5 segundos
    if time.time() - start_show_time <= SHOW_DURATION:
        for item in current_sequence:
            idx = item["idx"]
            color = item["color"]
            count = item["count"]
            center = centers[idx]
            draw_fruit_count(frame, center, r, color, count)
    # Después de 5 s ya no se dibujan

    cv2.imshow("Juego Frutal", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()



