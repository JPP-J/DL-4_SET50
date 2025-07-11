import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import os
import joblib
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight

# PyTorch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# Define a simple model using Embeddings for Collaborative Filtering
# class TorchModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim=32, output_dim=1):
#         super(TorchModel, self).__init__()

#         # Input Layer
#         self.fc1 = nn.Linear(input_dim, out_features=hidden_dim)  # Input layer (equivalent to Dense(6, input_dim=input_dim))

#         # Hidden Layer
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 1 hidden layer
#         self.dropout = nn.Dropout(0.2)  # Dropout after hidden layer

#         # Output Layer
#         self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer (equivalent to Dense(1))

#     def forward(self, X):
#         x = X.to(self.fc1.weight.device)
#         x = torch.relu(self.fc1(x))     # Input layer + ReLU activation
#         x = torch.relu(self.fc2(x))     # First hidden layer + ReLU
#         x = self.dropout(x)             # Apply Dropout
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)                 # final output

#         return x

class TorchModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(TorchModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # self.dropout1 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.dropout2 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        # self.dropout3 = nn.Dropout(0.3)


        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)



        self.fc6 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.fc1(x)))     # Layer 1 + BN + ReLU

        x = nn.functional.relu(self.bn2(self.fc2(x)))     # Layer 2 + BN + ReLU
        # x = self.dropout1(x)                              # Dropout for regularization
        x = nn.functional.relu(self.bn3(self.fc3(x)))     # Layer 3 + BN + ReLU
        x = self.dropout2(x)                              # Dropout for regularization
        x = nn.functional.relu(self.bn4(self.fc4(x)))     # Layer 4 + BN + ReLU
        # x = self.dropout3(x)                              # Dropout for regularization
        x = nn.functional.relu(self.bn5(self.fc5(x)))     # Layer 4 + BN + ReLU
        
        x = self.fc6(x)                                   # Output layer (raw logits)
        return x

class TorchClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dim, output_dim, epochs=50, lr=0.001, criteria='cross-ent', 
                 batch_size=16, val_size=0.2, patience=3.0 , debug=False):
        # self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_size = val_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patience = patience
        self.criteria = criteria
        self.pos_weight = None  # Optional: for handling class imbalance
        self.model = None  # initialize as None
        self.optimizer = None
        self.criterion = None
        self.history = None
        self.debug = debug

    @ staticmethod
    def _get_pos_weight(y):
        counter = Counter(y)
        print("Counter:", counter)
        print(counter[0], counter[1])
        num_neg = counter[0]
        num_pos = counter[1]
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
        print("Positive weight:", pos_weight)
        return torch.tensor(pos_weight, dtype=torch.float32) 
    
    def _get_criterion(self):
        # Criteria choices
        if self.criteria == 'cross-ent':
            self.criterion = nn.CrossEntropyLoss()      # label_smoothing=0.1
        elif self.criteria == 'MSE':
            self.criterion = nn.MSELoss()
        elif self.criteria == 'binary-logit':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        elif self.criteria == 'binary':
            self.criterion = nn.BCELoss()
        else:
            raise ValueError(f"Invalid criteria: {self.criteria}")
        
        return self.criterion
        # weight_tensor = torch.tensor([self.pos_weight], dtype=torch.float).to(self.device)  
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    
    def split_val(self, dataset):
        train_size = int((1-self.val_size) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        return train_dataset, val_dataset

    def fit(self, X, y):
        print(f"\nfiting process.....................................")

        # ─── Load and prepare data ──────────────────────────────────────────────────
        print(f'\nInitial load data....')
        print("Unique labels in full y:", np.unique(y))
        # print(f"Y: {Counter(y)}")
        
        if hasattr(X, "toarray"):
            X = X.toarray()

        input_dim = X.shape[1]  # Dynamically get input dimension for this fold
        self.input_dim = input_dim
        self.model = TorchModel(input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim).to(self.device)


        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)                      # Features
        y_tensor = torch.tensor(y.to_numpy(), dtype=torch.int64).to(self.device)         # Target
        # y_tensor = y_tensor.squeeze()  # Fix: Remove extra dimension (Shape: [16])

        # Create dataset and split into train/validation sets
        # Use DataLoader for batch training

        dataset = TensorDataset(X_tensor, y_tensor)
        train_dataset, val_dataset = self.split_val(dataset)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)  # can Reduced batch size
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)     # can Reduced batch size



        # ─── Inittal Parameters ──────────────────────────────────────────────────
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        # self.pos_weight = self._get_pos_weight(y).to(self.device)  # Calculate positive weight for handling class imbalance
        # weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        # class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        # self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.criterion = self._get_criterion()  # Get the criterion based on the specified criteria
        self.scaler = torch.amp.GradScaler()  # For mixed precision training

        torch.cuda.empty_cache()  # Clear GPU cache


        # ─── Initail for training  ──────────────────────────────────────────────────
        print(f'\nStart training data....')

        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        epochs = self.epochs

        print(f"Training for {epochs} epochs with batch size {self.batch_size}...\n")

        if self.debug:
            val_labels = []
            for _, batch_y in val_dataloader:
                val_labels.extend(batch_y.view(-1).tolist())
            print("Val label distribution:", Counter(val_labels))
        
        # batch_x, batch_y = next(iter(train_dataloader))

        # for i in range(100):
        #     self.optimizer.zero_grad()
        #     outputs = self.model(batch_x)
        #     loss = self.criterion(outputs.view(-1), batch_y.view(-1).float())
        #     loss.backward()
        #     self.optimizer.step()

        #     probs = torch.sigmoid(outputs.view(-1))
        #     preds = (probs > 0.5).float()
        #     acc = (preds == batch_y.view(-1)).float().mean().item()
        #     print(f"Epoch {i+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            correct_train = 0
            total_train = 0

            for batch_x, batch_y in train_dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).long()

                self.optimizer.zero_grad()  

                with torch.amp.autocast(device_type=self.device.type):  # Mixed precision
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)  # Fix: Ensure correct shape for BCEWithLogitsLoss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item()

                # ─── accuracy ──────────────────────────────────────────────────
                preds = outputs.argmax(dim=1)
                correct_train += (preds == batch_y).sum().item()
                total_train += batch_y.size(0)

                # print("Logits:", outputs.view(-1)[:5].detach().cpu().numpy())
                # print("Probs :", torch.sigmoid(outputs.view(-1))[:5].detach().cpu().numpy())
                # print("Preds :", preds[:5].detach().cpu().numpy())
                # print("Label :", batch_y[:5].cpu().numpy())

                # print("BATCH LABEL:", batch_y.view(-1)[:5])
                # print("BATCH LABEL dtype:", batch_y.dtype)
                # print("MODEL OUTPUT shape:", outputs.shape)
                # print("Model raw output logits:", outputs[:5])
                # print("Model raw output min/max:", outputs.min(), outputs.max())


            avg_train_loss = train_loss / len(train_dataloader)
            train_acc =  correct_train / total_train
            self.history["train_loss"].append(avg_train_loss)
            self.history["train_acc"].append(train_acc)

            # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            val_loss = 0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for batch_x, batch_y in val_dataloader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).long()

                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)

                    val_loss += loss.item()

                    # ─── accuracy ──────────────────────────────────────────────────
                    preds = outputs.argmax(dim=1)
                    correct_val += (preds == batch_y).sum().item()
                    total_val += batch_y.size(0)

                    # Inside validation loop
                    if total_val == 0:  # only once
                        print("VAL PRED:", preds[:10].cpu().numpy())
                        print("VAL TRUE:", batch_y[:10].cpu().numpy())
                    
                    if self.debug:
                        print("Logits:", outputs.view(-1)[:5].detach().cpu().numpy())
                        print("Probs :", torch.sigmoid(outputs.view(-1))[:5].detach().cpu().numpy())
                        print("Preds :", preds[:5].detach().cpu().numpy())
                        print("Label :", batch_y[:5].cpu().numpy())

            avg_val_loss = val_loss / len(val_dataloader)
            val_acc =  correct_val / total_val
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_acc"].append(val_acc)

            # Check for early stopping condition
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()  # Save the best model
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f'Early stopping triggered at epoch {epoch + 1}')
                    break

            # ─── PRINT EACH EPOCH──────────────────────────────────────────────────
            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        

        # Restore best model state
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        print(f'Finished training data....')
    
    def score(self, X_test, y_test):
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        y_pred = self.predict(X_test)
        print(np.unique(y_pred, return_counts=True))

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred ,average='macro', zero_division=1)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)
        report = classification_report(y_test, y_pred)
        cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return report, acc, precision, recall, f1, cm, cm_display

    def save_model(self, model_name:str=None):
        # Check if the folder already exists
        folder_name = "model"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        torch.save(self.model.state_dict(), f"model/{model_name}.pth")
        torch.save(self.model, f"model/{model_name}_complete.pth")
        torch.save(self.history, f"model/{model_name}_history.pth")

        print("Model and training history saved!")

    # def predict(self, X):
    #     if hasattr(X, "toarray"):
    #         X = X.toarray()

    #     self.model.eval()
    #     X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
    #     with torch.no_grad():
    #         outputs = self.model(X_tensor)
    #         probs = torch.sigmoid(logits.view(-1))  # Apply sigmoid to get probabilities
    #         preds = (probs > 0.5).float()  # Convert probabilities to binary predictions
    #         # (outputs.cpu().numpy() > 0.5).astype(int)  # Convert to 0 or 1
    #         return preds.cpu().numpy()
    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()

        self.model.eval()
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)          # shape (N, 3)

            # Option A ── raw argmax (fast, what metrics need)
            preds = logits.argmax(dim=1)           # shape (N,)

            # Option B ── if you also want probabilities:
            # probs = torch.softmax(logits, dim=1) # (N, 3)
            # preds = probs.argmax(dim=1)

            return preds.cpu().numpy()
        
    def predict_proba(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()

        """Returns probability scores for classification"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy()

    def plot_performance(self):

        plt.figure(figsize=(12, 4))

        # Get the length of training history
        train_loss_len = len(self.history['train_loss'])
        
        # Debug: Check what type and value we're working with
        # print(f"train_loss type: {type(self.history['train_loss'])}")
        # print(f"train_loss length: {train_loss_len}")
        
        # Create x_ticks_number based on the length
        if train_loss_len >= 100:
            x_ticks_number = range(0, train_loss_len, train_loss_len // 20)
        elif train_loss_len >= 20:
            x_ticks_number = range(0, train_loss_len, train_loss_len // 5)
        else:
            x_ticks_number = range(train_loss_len)


        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(x_ticks_number)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Training Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xticks(x_ticks_number)
        plt.legend()

        plt.tight_layout()
        plt.show()


class saved_model_usage:
    def __init__(self, path_model, path_his, path_pre):
        self.path_model = path_model
        self.path_his = path_his
        self.path_pre = path_pre
        self.model = None           # Optional initialization
        self.history = None     # Optional initialization
        self.preprocessor = None

    def load_model(self):
        # Load the saved model and training history
        self.model = torch.load(self.path_model)  # Load the model
        self.history = torch.load(self.path_his)  # Load the training history
        self.preprocessor = joblib.load(self.path_pre)


        # Optionally load the model's state_dict if needed
        # model.load_state_dict(torch.load("collaborative_filtering_param_model.pth"))

        print(f'\nModel Architecture:\n{self.model}')

        return self.model, self.history

    def preprocess_data_for_prediction(self, X):
        # Apply the same preprocessing steps that were applied during training
        X_processed = self.preprocessor.transform(X)  # This applies all preprocessing steps

        return X_processed
    def get_prediction(self, X):
        self.model.eval()
        X = self.preprocess_data_for_prediction(X)
        device = next(self.model.parameters()).device 

        # Set the model to evaluation mode
        # Making a prediction with the loaded model
        with torch.no_grad():                                            # No need to track gradients during inference
            input_data = torch.tensor(X, dtype=torch.float32, device=device)            # test data
            logits = self.model(input_data)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
    
        return  predictions

    def plot_saved_history(self):
        history = self.history

        # Plot the loss curves
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title("Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot the accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(history["train_acc"], label="Train Accuracy")
        plt.plot(history["val_acc"], label="Validation Accuracy")
        plt.title("Accuracy vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Print the final metrics after plotting
        print("\nFinal Metrics:")
        print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
        print(f"Final Val Accuracy: {history['val_acc'][-1]:.4f}")