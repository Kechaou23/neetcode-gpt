import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        max_z = np.max(z)
        exp_z = np.exp(z - max_z)
        sum = np.sum(exp_z)
        resu = exp_z / sum
        return np.round(resu, 4)
