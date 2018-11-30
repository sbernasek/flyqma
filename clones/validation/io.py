import dill as pickle


class Pickler:
    """ Methods for pickling an object instance. """

    def save(self, filepath):
        """ Save serialized instance. """
        with open(filepath, 'wb') as file:
            pickle.dump(self, file, protocol=0)

    @staticmethod
    def load(filepath):
        """ Save serialized instance. """
        with open(filepath, 'rb') as file:
            batch = pickle.load(file)
        return batch
