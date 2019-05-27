class LSH:
    def __init__(self):
        pass

    def add(self, q):
        return self.add_batch([q])

    def query(self, q):
        raise NotImplementedError()

    def add_batch(self, q_list):
        raise NotImplementedError()
