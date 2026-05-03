import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from model.config import ModelConfig

def plot_training(cfg: ModelConfig, train_acc_history: List[float], test_acc_history: List[float], plot_path: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_history, label="Train Accuracy", linewidth=2)
    plt.plot(test_acc_history, label="Test Accuracy", linewidth=2)

    colors = ['red', 'orange', 'green', 'blue', 'purple']
    if len(cfg.q_schedule) > len(colors):
        import numpy as np
        cmap = plt.get_cmap('rainbow')
        colors = cmap(np.linspace(0, 1, len(cfg.q_schedule)))

    # Find the lowest accuracy on the graph to anchor our text
    min_acc = min(min(train_acc_history), min(test_acc_history))

    for layer_idx, (q_sched, color) in enumerate(zip(cfg.q_schedule, colors)):
        y_offset = min_acc + (layer_idx * 0.05) 
        plt.axvline(q_sched._q_start, color=color, linestyle='-', alpha=0.6)
        plt.text(q_sched._q_start - 2, y_offset, f'L{layer_idx} Q-Start', rotation=90, color=color, fontsize=8)
        
        accumulated_epochs = q_sched._q_start
        for i, duration in enumerate(q_sched._epochs_per_bit[:-1]):
            accumulated_epochs += duration
            next_bits = max(q_sched._q_min_bits, q_sched._q_max_bits - i - 1)
            plt.axvline(accumulated_epochs, color=color, linestyle='--', alpha=0.3)
            plt.text(accumulated_epochs + 1, y_offset, f'{next_bits}b', rotation=90, color=color, fontsize=8)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy with Quantization Schedule")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_path)
    print(f"\nSaved training plot to {plot_path}")

    plt.show()