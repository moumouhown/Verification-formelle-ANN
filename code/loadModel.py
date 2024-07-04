import tensorflow as tf

class loadModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_model(self):
        return self.model
