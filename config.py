import os

class Config:
    def __init__(self):
        # Modelo de detección de manos
        self.model_path = os.path.join(os.path.dirname(__file__), 'models/hand_landmarker.task')
        
        # Parámetros del juego
        self.padding = 100
        self.game_time = 20     # Duración del juego en segundos
        self.circle_time = 1    # Duración de cada circulo azul antes de que desaparezca
        self.circle_time_radius = 15

config = Config()  # Instancia única
