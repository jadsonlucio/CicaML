from cicaML.processing import ProcessingMethod


class DataManager:
    def __init__(self, processing_methods, variables) -> None:
        self.processing_methods = processing_methods
        self.variables = variables

    def get_df(self, data):
        return data

    def apply_processing_methods(self, df, processing_methods):
        # Todo: finish this
        for processing_method in processing_methods:
            extra_params = processing_method.pop("params", {})
            method = processing_method.pop("method")
            if isinstance(method, str):
                method = ProcessingMethod.instances[method.lower()]
            df = method(df, **extra_params, **processing_method)

        return df

    def apply_processing_df(self, df):
        return self.apply_processing_methods(df, self.processing_methods)

    def apply_processing_variable(self, df, variable):
        processing_methods = ProcessingMethod.instances
        variable_processing = self.variables[variable]
        output_processing_methods = variable_processing["output_processing_methods"]
        for processing_method in output_processing_methods:
            method = processing_method["method"]
            if isinstance(method, str):
                method = processing_methods[method.lower()]

            params = processing_method.get("params", {})
            df = method(df, **params)

        return df

    def get_variable(self, variable, data, ignore_processing_df=False):
        if variable not in self.variables:
            raise Exception(
                f"variable {variable} not found, valid options are: {self.variables.keys()}"
            )
        df = self.get_df(data)
        if not ignore_processing_df:
            df = self.apply_processing_df(df)

        return self.apply_processing_variable(df, variable)

    @property
    def json(self):
        return {
            "processing_methods": self.processing_methods,
            "variables": self.variables,
        }
