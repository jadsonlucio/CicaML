from cicaML.pre_processing import PROCESSING_METHODS


class DataManager:
    def __init__(self, processing_methods, variables) -> None:
        self.processing_methods = processing_methods
        self.variables = variables

    def get_df(self, data):
        return data

    def apply_processing_df(self, df):
        for processing_method in self.processing_methods:
            params = processing_method.get("params", None)
            name = processing_method.get("name", None)
            replace = processing_method.get("replace", False)
            df = df.apply_processing_method(
                processing_method["column"],
                processing_method["method"],
                params=params,
                name=name,
                replace=replace,
            )
        return df

    def apply_processing_variable(self, df, variable):
        variable_processing = self.variables[variable]
        columns = list(set(variable_processing["columns"]))
        output_processing_methods = variable_processing["output_processing_methods"]
        use_df_input = variable_processing.get("use_df_input", False)
        data = df[columns] if use_df_input else df[columns].values
        for processing_method in output_processing_methods:
            method = processing_method["method"]
            if isinstance(method, str):
                method = PROCESSING_METHODS[method.lower()]

            params = processing_method.get("params", {})
            data = method(data, **params)

        return data

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
