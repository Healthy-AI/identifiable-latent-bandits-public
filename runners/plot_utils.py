import os

from matplotlib import pyplot as plt


def plot_loss_and_r2(history, ckpt_folder):
    """Plot training loss, validation loss, and validation R2 score."""
    plt.figure(figsize=(12, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss', color='tab:blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='tab:orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot validation R2 score
    plt.subplot(1, 2, 2)
    plt.plot(history['val_r2'], label='Validation R2', color='tab:green')
    plt.xlabel('Epochs')
    plt.ylabel('R2 Score')
    plt.title('Validation R2 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_folder, 'loss_and_r2_plot.pdf'))
    plt.clf()
