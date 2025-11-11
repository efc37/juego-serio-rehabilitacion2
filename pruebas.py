import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # radio
    r = int(min(w / 16, h / 9))

    # posiciones centradas
    x_margin = int(w * 0.18)
    y_margin = int(h * 0.20)
    x_positions = np.linspace(x_margin, w - x_margin, 4, dtype=int)
    y_positions = np.linspace(y_margin, h - y_margin, 3, dtype=int)

    # --- líneas horizontales ---
    for y in y_positions:
        for i in range(len(x_positions) - 1):
            x1 = x_positions[i] + r   # empieza después del borde del círculo
            x2 = x_positions[i + 1] - r  # termina antes del siguiente círculo
            cv2.line(frame, (x1, y), (x2, y), (200, 200, 200), 2)

    # --- líneas verticales ---
    for x in x_positions:
        for j in range(len(y_positions) - 1):
            y1 = y_positions[j] + r   # empieza después del borde del círculo
            y2 = y_positions[j + 1] - r  # termina antes del siguiente
            cv2.line(frame, (x, y1), (x, y2), (200, 200, 200), 2)

    # --- círculos ---
    for y in y_positions:
        for x in x_positions:
            cv2.circle(frame, (x, y), r, (0, 0, 0), 2)

    cv2.imshow("Juego Frutal", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
