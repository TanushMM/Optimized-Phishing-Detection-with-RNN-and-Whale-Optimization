from data_preprocessing import load_and_preprocess_data
from model import build_rnn_model
from train import train_model_with_WOA, evaluate_model
from utils import plot_training_history

data_path = ""
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(data_path)

input_shape = (X_train.shape[1], X_train.shape[2])

model = build_rnn_model(input_shape)

model, history = train_model_with_WOA(model, X_train, y_train, X_val, y_val)

evaluate_model(model, X_test, y_test)

if history:
    plot_training_history(history)
