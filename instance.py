import pickle

def read_list(filename):
    # for reading also binary mode is important
    with open(filename, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

def example_instance():
    # Example instance: 4 items with weights [2, 5, 4, 7], capacity 10, and 3 bins
    example_instance = [[2, 5, 4, 7], 10, 3, 4]
    return example_instance