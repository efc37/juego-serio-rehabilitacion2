import os

class Config:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'models/pose_landmarker_full.task') #pose completa
        self.padding = 100
        self.game_time = 90     # Duración del juego en segundos
        self.frute_time = 5  # Duración de cada circulo azul antes de que desaparezca
        #self.circle_time_radius = 15 NO SE LO QUE ES

config = Config()  # Instancia única