import cv2
import threading
import queue
import numpy as np

class ThreadStream:
    def __init__(self, source, queue_size: int = 3, drop_old: bool | None = None):
        self.source = source
        self.stream = cv2.VideoCapture(source)

        if not self.stream.isOpened():
            raise ValueError(f"Stream video not opened: {source}")

        if drop_old is None:
            drop_old = isinstance(source, int) or str(source).startswith(("rtsp", "http"))
        self.drop_old = drop_old

        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.finished = False
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                break

            if self.drop_old:
                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put(frame)
            else:
                while not self.stopped:
                    try:
                        self.q.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        self.finished = True

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def more(self) -> bool:
        return (not self.finished) or (not self.q.empty())

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.stream.release()