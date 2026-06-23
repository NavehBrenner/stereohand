"""Serve several webcams as MJPEG-over-HTTP streams — the WSL2 two-camera bridge.

WSL2's kernel has no UVC driver, so host webcams never appear as devices inside WSL. Run
THIS on **Windows** (where the cameras live) and have the WSL-side stereohand open them by
URL instead of a device index — one path per camera::

    # on Windows (needs: pip install opencv-python):
    python scripts/stream_webcams.py --cameras 0 1      # serves /0 and /1 on :8080

    # in WSL (Python):
    from stereohand import StereoHandTracker, StereoCalibration
    calib = StereoCalibration.load("stereo_calib.json")
    tracker = StereoHandTracker.open(
        calib,
        left="http://<windows-host>:8080/0",
        right="http://<windows-host>:8080/1",
    )

Finding ``<windows-host>`` from WSL:
- mirrored networking (``networkingMode=mirrored`` in ``.wslconfig``): use ``localhost``.
- default NAT networking: the default-route gateway —
  ``ip route show default | awk '{print $3}'``.

First run pops a Windows Firewall prompt — allow it on private networks, or WSL can't
connect. Stdlib + OpenCV only (no stereohand imports): it runs on the Windows host, outside
any venv, so it logs with ``print``.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

_BOUNDARY = "frame"


def parse_camera_index(path: str, allowed: set[int]) -> int | None:
    """Map a request path (``/0``, ``/1/video``) to a whitelisted camera index, or None."""
    first = path.strip("/").split("/", 1)[0]
    if not first.isdigit():
        return None
    index = int(first)
    return index if index in allowed else None


def _make_handler(allowed: set[int], jpeg_quality: int) -> type[BaseHTTPRequestHandler]:
    class MJPEGHandler(BaseHTTPRequestHandler):
        def _multipart_headers(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
            self.end_headers()

        def do_HEAD(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            # Firewall/reachability pre-flight: headers only, no camera open.
            if parse_camera_index(self.path, allowed) is None:
                self.send_error(404)
                return
            self._multipart_headers()

        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            camera = parse_camera_index(self.path, allowed)
            if camera is None:
                self.send_error(404, f"unknown camera path {self.path!r}")
                return
            # DirectShow opens in <1s; the default MSMF backend stalls ~10s per cam on Windows.
            capture = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
            if not capture.isOpened():
                self.send_error(500, f"could not open camera {camera}")
                return
            self._multipart_headers()
            print(f"camera {camera}: client connected {self.client_address[0]}")
            try:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
                    if not ok:
                        continue
                    payload = jpeg.tobytes()
                    self.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode())
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                print(f"camera {camera}: client disconnected {self.client_address[0]}")
            finally:
                capture.release()

        def log_message(self, *args: object) -> None:
            pass  # quiet the default per-request access log

    return MJPEGHandler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cameras", type=int, nargs="+", default=[0, 1], help="camera indices to serve"
    )
    parser.add_argument("--port", type=int, default=8080, help="port to serve on (default 8080)")
    parser.add_argument(
        "--jpeg-quality", type=int, default=80, help="JPEG quality 1-100 (default 80)"
    )
    args = parser.parse_args()

    allowed = set(args.cameras)
    handler = _make_handler(allowed, args.jpeg_quality)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    for index in sorted(allowed):
        print(f"camera {index}: http://0.0.0.0:{args.port}/{index}")
    print("Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
