import io
import numpy as np

class StepResult:
    is_terminal:bool
    reward:float
    state:np.ndarray
    image_buffer:io.BytesIO
