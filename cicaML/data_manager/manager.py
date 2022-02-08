from typing import List, Union
from pandas import DataFrame

from cicaML.processing import PROCESSING_METHODS
from cicaML.types.base import EmptyDict
from cicaML.types.ml import (
    DataManagerVariable,
    DataManagerVariables,
    ProcessingMethodDict,
)


class DataManager:
    def __init__(
        self,
        processing_methods: List[ProcessingMethodDict],
        variables: Union[DataManagerVariables, EmptyDict],
    ) -> None:
        self.processing_methods = processing_methods
        self.variables = variables

    def apply_processing_methods(
        self, input_data, processing_methods: List[ProcessingMethodDict]
    ):
        for processing_method in processing_methods:
            extra_params = processing_method.pop("params", {})
            method = processing_method.pop("method")
            if isinstance(method, str):
                method = PROCESSING_METHODS[method.lower()]
            input_data = method(input_data, **extra_params, **processing_method)

        return input_data

    def apply_processing_df(self, df: DataFrame):
        return self.apply_processing_methods(df, self.processing_methods)

    def apply_processing_variable(self, input_data, variable: str):
        variable_processing: DataManagerVariable = self.variables[variable]
        output_processing_methods = variable_processing["output_processing_methods"]
        for processing_method in output_processing_methods:
            method = processing_method["method"]
            if isinstance(method, str):
                method = PROCESSING_METHODS[method.lower()]

            params = processing_method.get("params", {})
            input_data = method(input_data, **params)

        return input_data

    def get_variable(self, variable: str, df: DataFrame, ignore_processing_df=False):
        if variable not in self.variables:
            raise Exception(
                f"Variable {variable} not found, valid options are: {self.variables.keys()}"
            )
        if not ignore_processing_df:
            df = self.apply_processing_df(df)

        return self.apply_processing_variable(df, variable)

    @property
    def json(self):
        return {
            "processing_methods": self.processing_methods,
            "variables": self.variables,
        }
