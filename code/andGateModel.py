import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import onnx
from tensorflow.keras.models import load_model
import tf2onnx

def andy(X):
    """Boolean AND gate."""
    return float(X[0] == 1 and X[1] == 1)

def mk_andy_data(epsilon):
    """Create data for training an AND gate."""
    T = np.linspace(1-epsilon, 1+epsilon, num=100)
    F = np.linspace(0-epsilon, 0+epsilon, num=100)
    D = np.append(T, F)
    X = np.array([(x1, x2) for x1 in D for x2 in D])
    Y = np.array([andy(x) for x in X])
    return (X, Y)

class andGateModel:
    
    def __init__(self, model_path=None):
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.model = self.build_model()

    def build_model(self, epsilon=0):
        x_train, y_train = mk_andy_data(epsilon=epsilon)
        model = keras.Sequential([
            keras.layers.Dense(3, activation='relu', input_shape=(2,)),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10)
        # Save the trained model
        model.save("andGateModel.keras")

        # Export the TensorFlow model as a saved model
        tf.saved_model.save(model, "andGateModel")
        return model
    
    def load_model(self, model_path):
        return load_model(model_path)
        
    def get_model(self):
        return self.model

# Create an instance of andGateModel
and_gate_model_instance = andGateModel()
model = and_gate_model_instance.get_model()
