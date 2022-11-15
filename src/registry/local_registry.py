import io
import os

from .registry import Registry

class LocalRegistry(Registry):
    def __init__(self, eval_root: str) -> None:
        self.eval_root = eval_root if eval_root[0] == "." else f".{eval_root}"

    def save_run_history(self, algorithm:str, run_history) -> None:
        episode_list = list(range(1, run_history.episodes + 1))

        self.write_plot(
            x_list=episode_list,
            y_lists=[run_history.total_rewards, run_history.max_rewards],
            plot_labels=["Reward", "Max reward"],
            x_label="Episode",
            y_label="Reward",
            title=f"Training: Rewards ({algorithm})",
            filename=f"plot-training-{algorithm}-rewards.png"
        )

        if isinstance(run_history.epsilon, list) and len(run_history.epsilon) > 0:
            self.write_plot(
                x_list=episode_list,
                y_lists=[run_history.epsilon],
                plot_labels=["Epsilon"],
                x_label="Episode",
                y_label="Epsilon",
                title="Epsilon Decay",
                filename=f"plot-training-{algorithm}-epsilon.png"
            )

    def write_bytes(self, root:str, filename:str, buffer:io.BytesIO):
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, filename)
        with open(path, mode="wb") as fd:
            fd.write(buffer.getvalue())

    def read_bytes(self, root:str, filename:str) -> io.BytesIO:
        file_path = os.path.join(root, filename)
        assert os.path.exists(file_path), f"Error: the file path {file_path} could not be found"
        reader = open(file_path, mode="rb")
        buffer = io.BytesIO(reader.read())
        return buffer
