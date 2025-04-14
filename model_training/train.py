import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from WOA import WhaleOptimization
from model import build_rnn_model

def fitness_function(params, input_shape, X_train, y_train, X_val, y_val):
    """Fitness function for WOA."""
    learning_rate, batch_size = params
    batch_size = int(batch_size)
    learning_rate = max(1e-5, min(learning_rate, 1e-2))
    batch_size = max(16, min(batch_size, 128))

    print(f"🔹 Testing Hyperparameters -> Learning Rate: {learning_rate:.6f}, Batch Size: {batch_size}")

    try:
        model = build_rnn_model(input_shape)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(factor=0.2, patience=2, min_lr=1e-6)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )

        val_acc = max(history.history.get('val_accuracy', [0]))
        return -val_acc  # Negative for minimization
    except Exception as e:
        print(f"❌ Error in fitness function: {e}")
        return float("inf")

def train_model_with_WOA(model, X_train, y_train, X_val, y_val, epochs=20, max_iterations=2):
    """Trains model with WOA hyperparameter optimization."""
    input_shape = (X_train.shape[1], X_train.shape[2])
    constraints = [(1e-5, 1e-2), (16, 128)]
    history_log = []

    try:
        woa = WhaleOptimization(
            opt_func=lambda params: fitness_function(params, input_shape, X_train, y_train, X_val, y_val),
            constraints=constraints,
            nsols=5,
            b=1.5,
            a=2.0,
            a_step=0.1,
            maximize=False
        )
    except Exception as e:
        print(f"❌ WOA Initialization Failed: {e}")
        return None, None

    print("\n🔵 Starting Whale Optimization...\n")

    for iteration in range(max_iterations):
        try:
            print(f"🔄 Iteration {iteration + 1}/{max_iterations}...")
            woa.optimize()
            best_fitness, best_hyperparams = woa.get_best_solution()

            if best_fitness == float("inf"):
                print("⚠️ Skipping invalid solution...")
                continue

            history_log.append(-best_fitness)

            print(f"🟢 Best Accuracy so far: {-best_fitness:.6f} | Hyperparameters: {best_hyperparams}")
        except Exception as e:
            print(f"❌ Error in WOA iteration: {e}")

    try:
        best_lr, best_batch_size = woa.get_best_solution()[1]
        best_batch_size = int(best_batch_size)

        print(f"\n✅ Final Hyperparameters - LR: {best_lr:.6f}, Batch Size: {best_batch_size}")

        optimizer = Adam(learning_rate=best_lr)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=best_batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        model.save('./model.h5')

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(history_log) + 1), history_log, marker='o', linestyle='-', color='blue')
        plt.xlabel("WOA Iteration")
        plt.ylabel("Validation Accuracy")
        plt.title("WOA Optimization Progress")
        plt.grid(True)
        plt.show()

        return model, history

    except Exception as e:
        print(f"❌ Final training error: {e}")
        return None, None

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    if model is None:
        raise ValueError("Model not trained yet.")

    try:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
        print(f"\n🔴 Final Test Loss: {test_loss:.4f}")
        print(f"🟢 Final Test Accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None, None
