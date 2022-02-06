def identity(x):
    return x


def mix(x):
    return len(set(x))


def validate_type(x, type_):
    if not isinstance(x, type_):
        raise TypeError(f"The input must be of type {type_}, not {type(x)}")

    return True
