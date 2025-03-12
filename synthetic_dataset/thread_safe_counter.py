from collections import Counter
from threading import Lock

class ThreadSafeCounter:
    def __init__(self):
        self._counter: Counter = Counter()
        self._lock = Lock()

    def update(self, *args, **kwargs):
        with self._lock:
            self._counter.update(*args, **kwargs)

    def __getitem__(self, key):
        with self._lock:
            return self._counter[key]

    def values(self):
        with self._lock:
            return self._counter.values()
    
    def __len__(self):
        with self._lock:
            return len(self._counter)

    def __setitem__(self, key, value):
        with self._lock:
            self._counter[key] = value
    
    def values(self):
        with self._lock:
            return self._counter.values()
