from typing import List, Tuple, TypedDict, Union

from .base import GenericList

TrainData = Tuple[GenericList, GenericList, GenericList, GenericList]


class ProcessingMethodDict(TypedDict):
    name: Union[str, None]
    method: str
    params: Union[dict, None]
    column: str
    replace: bool


class DataManagerVariable(TypedDict):
    columns: list
    output_processing_methods: List[ProcessingMethodDict]
    use_df_input: bool


class DataManagerVariables(TypedDict):
    x: DataManagerVariable
    y: DataManagerVariable
    train: DataManagerVariable
