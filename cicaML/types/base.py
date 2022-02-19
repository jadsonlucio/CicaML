import numpy as np
from typing import Sequence, TypedDict, Union

GenericList = Union[list, np.ndarray, tuple, Sequence]


class EmptyDict(TypedDict):
    pass
