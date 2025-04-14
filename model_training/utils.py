import matplotlib.pyplot as plt

def plot_training_history(history):
    """ Plots training and validation loss & accuracy """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(history.history.get('accuracy', []), label='Train Accuracy')  # Handling cases where accuracy might not be tracked
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
