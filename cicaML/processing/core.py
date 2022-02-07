from numpydoc.docscrape import NumpyDocString


class ProcessingMethod:
    instances = {}

    def __init__(self, method, input_type, output_type, name=None):
        self.name = name or self.method.__name__

        if method.__doc__ is None:
            raise Exception(f"Processing method {self.name} has no docstring")

        self.method = method
        self.input_type = input_type
        self.output_type = output_type
        self.doc = NumpyDocString(self.method.__doc__)

        self.instances[self.name] = self

    def validate(self, *args, **kwargs):
        params = [parameter.name for parameter in self.doc["Parameters"]]

        for key in kwargs:
            if key not in params:
                raise KeyError(f"{key} is not a valid parameter for {self.name}")

    def __call__(self, *args, **kwargs):
        self.validate(*args, **kwargs)
        return self.method(*args, **kwargs)
