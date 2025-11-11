#Sara y Elena

import sys
import cv2
import mediapipe as mp
import numpy as np
import time
from config import config


# Configuración de la tarea
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=config.model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)

# L2 Norm entre la mano y el punto objetivo (azul) 
def calculate_distance(point1, point2) -> float:
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return distance

# Crea una nueva capa y una posición random para el objetivo ---- NUEVO NIVEL
'''def new_circle(padding) -> None:
    global circle_layer, random_circle_position
    circle_layer =  np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)
    random_circle_position = (np.random.randint(padding, circle_layer.shape[1]-padding), np.random.randint(padding, circle_layer.shape[0]-padding))
'''
# Inicializar landmarker
with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)  # Abrir cámara
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # seguridad si la cámara no da fps
        fps = 30

    mp_pose = mp.solutions.pose

    frame_ms = int(1000 / fps)

    # Inicializar valores
    timestamp = start_time = end_time = sum_t = counter = final_counter = 0

    blue_back = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)
    blue_back[:] = (255, 0, 0)

    new_circle(config.padding)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Voltear frame horizontalmente
        frame = cv2.flip(frame, 1)

        # Dibuja el circulo random en el frame
        cv2.circle(circle_layer, random_circle_position, 20, (255, 255, 255), -1)
        h, w = circle_layer.shape[:2]
        frame = np.where(circle_layer != 0, blue_back, frame)

        # Variables de tiempo
        t = end_time - start_time
        sum_t += t
        start_time = time.time()

        # Convertir frame a formato MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Ejecutar detección
        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += frame_ms

        # Dibujar landmarks si los hay
        if result.pose_landmarks:
            for person_landmarks in result.pose_landmarks:  # para mantener tu estructura
                h, w, _ = frame.shape

                # Dibuja líneas entre los landmarks
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = person_landmarks[start_idx]
                    end = person_landmarks[end_idx]

                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)

                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Dibuja los puntos (landmarks)
                for idx, landmark in enumerate(person_landmarks):
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                landmarks_points = [(landmark.x, landmark.y) for landmark in [person_landmarks[19], person_landmarks[20]]]
                for landmark in landmarks_points:
                    h, w, _ = frame.shape
                    x, y = int(landmark[0] * w), int(landmark[1] * h)
                    if calculate_distance(landmark, (random_circle_position[0]/w, random_circle_position[1]/h)) < 0.05:
                        counter += 1
                        sum_t = 0
                        new_circle(config.padding)
                        break

        if sum_t > config.circle_time:
            sum_t = 0
            new_circle(config.padding)
        else:
            radius = int((20 + config.circle_time) - (config.circle_time_radius * (sum_t - config.circle_time)) / config.circle_time)
            cv2.circle(frame, random_circle_position, radius, (255, 255, 255), 2)

        end_time = time.time()

        # Mostrar tiempo restante de juego y puntos obtenidos
        cv2.putText(frame, f'Time: {config.game_time:.2f} - Points: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        config.game_time -= t 
        
        if config.game_time <= 0:
            frame = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)
            cv2.putText(frame, f'Score: {final_counter} points', (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Press Enter to continue ...', (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Press ESC to exit', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 2)

            # Tecla Enter para jugar
            if cv2.waitKey(1) & 0xFF == 13:
                config.game_time = 20
                counter = 0
            
            # Tecla ESC para salir
            if cv2.waitKey(1) & 0xFF == 27:
                sys.exit(0)
        else:
            final_counter = counter

        # Mostrar resultado
        cv2.imshow("Pose Landmarker", frame)

        # Tecla ESC para salir
        if cv2.waitKey(1) & 0xFF == 27:
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()