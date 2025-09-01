import threading
import queue
from concurrent.futures import Future




class ModelServer(threading.Thread):
    def __init__(self, model, device, q):
        super().__init__(daemon=True)

        self.model = model
        self.device = device
        self.q = q
    
    def submit(self, inputs):
        future = Future()
        self.q.put((inputs, future))
        return future

    def run(self):
        # start checking queue for things to execute
        while not (self.stop_flag.is_set() and self.q.empty()):
            try:
                inputs, future = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            inputs.to(self.device)
            forward = self.model(**inputs, return_prob=True)
            future.set_result(forward)

    def stop(self):
        self.stop_flag.set()