import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # efecto espejo (más natural)
    frame = cv2.flip(frame, 1)

    # tamaño del frame
    h, w = frame.shape[:2]

    # radio un poco más pequeño
    r = int(min(w / 16, h / 9))

    # posiciones (4 columnas x 3 filas), centradas mejor visualmente
    x_margin = int(w * 0.18)
    y_margin = int(h * 0.20)
    x_positions = np.linspace(x_margin, w - x_margin, 4, dtype=int)
    y_positions = np.linspace(y_margin, h - y_margin, 3, dtype=int)

    # --- líneas horizontales ---
    for y in y_positions:
        for i in range(len(x_positions) - 1):
            cv2.line(frame, (x_positions[i], y), (x_positions[i + 1], y), (200, 200, 200), 2)

    # --- líneas verticales ---
    for x in x_positions:
        for j in range(len(y_positions) - 1):
            cv2.line(frame, (x, y_positions[j]), (x, y_positions[j + 1]), (200, 200, 200), 2)

    # --- círculos transparentes (solo borde negro) ---
    for y in y_positions:
        for x in x_positions:
            cv2.circle(frame, (x, y), r, (0, 0, 0), 2)

    cv2.imshow("Juego Frutal", frame)

    # salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
