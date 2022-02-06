from .core import ProcessingMethod


def processing_method(name, input_type, output_type):
    def wrapper(func):
        processing_method = ProcessingMethod(func, input_type, output_type, name)
        return processing_method

    return wrapper
