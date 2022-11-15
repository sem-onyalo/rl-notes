import io

import matplotlib.pyplot as plt

class Registry:
    def __init__(self, eval_root:str) -> None:
        self.eval_root = eval_root

    def write_plot(self, x_list, y_lists, plot_labels, x_label, y_label, title, filename):
        for i in range(0, len(y_lists)):
            plt.plot(x_list, y_lists[i], label=plot_labels[i])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()

        buffer = io.BytesIO()
        plt.savefig(buffer)
        plt.close()

        self.write_bytes(self.eval_root, filename, buffer)

    def save_model(self, filename:str, buffer:io.BytesIO) -> None:
        self.write_bytes(self.eval_root, filename, buffer)

    def load_model(self, filename:str) -> io.BytesIO:
        return self.read_bytes(self.eval_root, filename)

    def save_run_history(self, algorithm:str, run_history) -> None:
        pass

    def write_bytes(self, root:str, filename:str, buffer:io.BytesIO):
        pass

    def read_bytes(self, root:str, filename:str) -> io.BytesIO:
        pass
