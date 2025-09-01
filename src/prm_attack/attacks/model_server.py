import threading
import queue


class ModelServer(threading.Thread):
    def __init__(self, model, device, q, response_qs):
        super().__init__(daemon=True)

        self.model = model
        self.device = device
        self.q = q
        self.response_qs = response_qs

        self.stop_flag = threading.Event()
    
    def run(self):
        # start checking queue for things to execute
        while not (self.stop_flag.is_set() and self.q.empty()):
            try:
                rank, inputs = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            inputs.to(self.device)
            forward = self.model(**inputs, return_prob=True)
            self.response_qs[rank].put((forward.rewards[inputs.data["reward_flags"].bool()]).tolist())

    def stop(self):
        self.stop_flag.set()