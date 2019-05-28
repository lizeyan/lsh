from timeit import default_timer as timer


class Timer:
    def __init__(self):
        self.tic = 0
        self.toc = 0

    @property
    def elapsed_time(self):
        return self.toc - self.tic

    def __enter__(self):
        self.tic = timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc = timer()
        return self
