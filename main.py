import cv2
import time
from loguru import logger

from core import Inferencer
from utils import Visualizer, ThreadStream

model = Inferencer("models/person.pt", 'cuda:0', conf_thresh=0.25)
visualizer = Visualizer()

def run(source):
    logger.info(f"Open stream: {source}")
    start = time.time()
    stream = ThreadStream(source=source).start()

    try:
        while stream.more():
            t0 = time.time()

            frame = stream.read()
            if frame is None:
                continue

            results = model.predict(frame)

            vis = frame
            for det in results:
                vis = visualizer.draw_detection(vis, det)

            fps = 1.0 / max(time.time() - t0, 1e-6)
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Inference", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        logger.info(f"System stopped in {time.time() - start:.4f}s")

if __name__ == "__main__":
    source = "tests/test.mp4"
    run(source)