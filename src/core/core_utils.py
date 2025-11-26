import math

counter = []

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def name_generator(cortex, name=None):
    """Returns a name as a string."""
    counter.append(0)
    if name is None:
        return f"{cortex.name}_{len(counter)}"
    else:
        return str(name)
