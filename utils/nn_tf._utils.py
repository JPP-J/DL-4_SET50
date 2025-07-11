import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import layers, models
import joblib
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report


# Feedforward neural network FFNN (input → hidden layers → output)
# Define the Custom Keras Classifier
class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, epochs=50):
        self.input_dim = input_dim
        self.model = self.build_model()
        self.history = None  # Initialize history attribute
        self.epochs = epochs
        self.classes_ = None  # Initialize classes_ attribute


    def build_model(self):
        model = Sequential([
            # Input Layer
            # Input(shape=(self.input_dim,)),                             # Input optional
            Dense(6, input_dim=self.input_dim, activation='relu'),  # Input

            # First Hidden Layer (Dense layer/Fully connected layer)
            Dense(8, activation='relu'),    # ReLU activation: Common for hidden layers
            Dropout(0.2),

            # Output Layer
            Dense(1, activation='sigmoid')  # "sigmoid": binary, "softmax": multiclass, output
        ])

        # Loss function: 'binary_crossentropy' for binary classification
        # Loss function: 'sparse_categorical_crossentropy' for multi-class classification
        # Optimizer: 'adam' is an efficient optimizer.
        # optimizer = SGD(learning_rate=self.learning_rate, momentum=self.momentum)
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # Store the unique classes
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,                # Number of epochs to wait for improvement
            restore_best_weights=True   # Restore the best weights after stopping
        )
        self.history = self.model.fit(
            X, y,
            epochs= self.epochs,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        return self

    def predict(self, X):
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def score(self, X, y):
        y_pred = self.model.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else (y_pred > 0.5).astype(int)
        return accuracy_score(y, y_pred_classes)

    def get_summary(self):
        self.model.summary()

    def get_history(self):
        return self.history.history

    def plot_model(self):       # Only have Graphviz Software
        # Visualize the model
        plot_model(self.model, to_file='data/ann_model_structure.png', show_shapes=True, show_layer_names=True)

    def plot_history(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(range(len(self.history.history['loss'])))
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(self.history.history['accuracy'])))
        plt.legend()

        plt.tight_layout()
        plt.show()

def load_model(file_path): # to thrive data after trained
    # Load the model
    model = models.load_model(file_path)

    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)
    model.compile(
        optimizer=optimizer,            # or any other optimizer using
        loss='binary_crossentropy',     # or loss function
        metrics=['accuracy']            # or other metrics
    )

    # model summary
    print(model.summary())

    # List all layers
    for layer in model.layers:
        print(f"Layer: {layer.name}, Type: {type(layer).__name__}")

    return model

def load_history(file_path):
    history_dict = joblib.load(file_path)
    return history_dict

def plot_saved_history(file_path):
    saved_history = load_history(file_path)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(saved_history['loss'], label='Training Loss')
    plt.plot(saved_history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(saved_history['accuracy'], label='Training Accuracy')
    plt.plot(saved_history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()