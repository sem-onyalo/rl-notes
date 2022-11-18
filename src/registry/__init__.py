import io
import os

import matplotlib.pyplot as plt

from .local_registry import LocalRegistry
from .registry import Registry

def plot_training_metrics(
    algorithm:str, 
    episodes:int, 
    total_rewards:list, 
    max_rewards:list, 
    epsilon:list=None) -> None:

    episode_list = list(range(1, episodes + 1))

    write_plot(
        x_list=episode_list,
        y_lists=[total_rewards, max_rewards],
        plot_labels=["Reward", "Max reward"],
        x_label="Episode",
        y_label="Reward",
        title=f"Training: Rewards ({algorithm})",
        filename=f"plot-training-{algorithm}-max-rewards.png"
    )

    if isinstance(epsilon, list) and len(epsilon) > 0:
        write_plot(
            x_list=episode_list,
            y_lists=[epsilon],
            plot_labels=["Epsilon decay"],
            x_label="Episode",
            y_label="Epsilon",
            title="Epsilon Decay",
            filename=f"plot-training-{algorithm}-epsilon-decay.png"
        )

def write_plot(x_list, y_lists, plot_labels, x_label, y_label, title, filename):
    for i in range(0, len(y_lists)):
        plt.plot(x_list, y_lists[i], label=plot_labels[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer)
    plt.close()

    write_bytes("./eval", filename, buffer)

def write_bytes(root:str, filename:str, buffer:io.BytesIO):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, filename)
    with open(path, mode="wb") as fd:
        fd.write(buffer.getvalue())

def read_bytes(root:str, filename:str) -> io.BytesIO:
    file_path = os.path.join(root, filename)
    assert os.path.exists(file_path), f"Error: the file path {file_path} could not be found"
    buffer = open(file_path, mode="rb")
    return buffer