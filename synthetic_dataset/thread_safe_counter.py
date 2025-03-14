from threading import Lock

class ThreadSafeCounter:
    def __init__(self, initial=0):
        self.count = initial
        self._lock = Lock()
    
    def add(self, increment):
        with self._lock:
            self.count += increment
    
    def set_val(self, val):
        with self._lock:
            self.count = val
    
    @property
    def value(self):
        return self.count
