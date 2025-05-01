import time

class TimingWrapper:
    """Base class for timing wrappers."""
    def __init__(self, iterator, name):
        self.iterator = iterator
        self.name = name
        self.iter_obj = iter(self.iterator)

    def __getattr__(self, attr):
        # Delegate attribute access to the wrapped iterator if not found on self
        return getattr(self.iterator, attr)

    def __iter__(self):
        return self

    def __next__(self):
        start_time = time.time()
        try:
            item = next(self.iter_obj)
            end_time = time.time()
            print(f"  [{self.name}] next() took: {end_time - start_time:.6f} seconds")
            return item
        except StopIteration:
            end_time = time.time()
            print(f"  [{self.name}] StopIteration after: {end_time - start_time:.6f} seconds")
            raise
        except Exception as e:
            end_time = time.time()
            print(f"  [{self.name}] Error after: {end_time - start_time:.6f} seconds")
            raise e

# You might need specific wrappers if __init__ differs significantly
# or if you need to wrap specific methods other than __next__

class ArrowFileIteratorWrapper(TimingWrapper):
    def __init__(self, iterator):
        super().__init__(iterator, "ArrowFileIterator")
    def create_iter(self):
        return self.iterator.create_iter()

class PreprocessIteratorWrapper(TimingWrapper):
    def __init__(self, iterator):
        super().__init__(iterator, "PreprocessIterator")
    def create_iter(self):
        return self.iterator.create_iter()

class SequenceIteratorWrapper(TimingWrapper):
    def __init__(self, iterator):
        super().__init__(iterator, "SequenceIterator")
    def create_iter(self):
        return self.iterator.create_iter()

class SamplingIteratorWrapper(TimingWrapper):
    def __init__(self, iterator):
        super().__init__(iterator, "SamplingIterator")
    def create_iter(self):
        return self.iterator.create_iter()

class PackingIteratorWrapper(TimingWrapper):
    def __init__(self, iterator):
        super().__init__(iterator, "PackingIterator")
    def create_iter(self):
        return self.iterator.create_iter()
