from multiprocessing import Value

class ProcessSafeCounter:
    def __init__(self, initial=0):
        self._count = Value("i", initial)
    
    def change(self, increment):
        with self._count.get_lock():
            self._count.value += increment
    
    def set_val(self, val):
        with self._count.get_lock():
            self._count.value = val
    
    @property
    def value(self):
        return self._count.value
