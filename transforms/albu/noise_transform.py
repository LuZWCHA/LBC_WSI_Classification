
from typing import Tuple
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from ..noises import RandomNoise

class JNoise(ImageOnlyTransform):
    
    def __init__(self, scale=-1, always_apply: bool = False, p: float = 0.2):
        super().__init__(always_apply, p)
        self.scale = scale
        self.random_noise = RandomNoise(scale=scale)
    
    "A transform handler for multiple `Albumentation` transforms"
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return self.random_noise(img)
    
    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ["scale"]
    
    

    