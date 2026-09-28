"""Apple freshness checker: runs on the laptop, no internet or GPU needed.

Live camera mode (default):
  SPACE = check the side of the apple facing the camera
          (turn the apple and press SPACE again to check more sides)
  N     = new apple (clears the result)
  Q     = quit
The verdict uses the WORST side seen so far: one rotten side means the apple is rotten.

Test on photos instead of the camera:
  python apple_checker.py --image some_apple.jpg other.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from common import DEFAULT_ROI, center_square, draw_box, open_camera, put_text, speak

HERE = Path(__file__).resolve().parent

VERDICT_STYLE = {  # BGR colours
    "FRESH":  ((40, 170, 40),  "Fresh",  "This apple looks fresh."),
    "ROTTEN": ((30, 30, 220),  "Rotten", "This apple looks rotten. Do not eat it."),
    "UNSURE": ((0, 190, 240),  "Not sure", "I am not sure. Please ask someone to check this apple."),
}


class AppleModel:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        onnx_path, info_path = model_dir / "apple_model.onnx", model_dir / "model_info.json"
        if not onnx_path.exists() or not info_path.exists():
            raise SystemExit(f"Model not found. Put apple_model.onnx and model_info.json in: {model_dir}")
        self.info = json.loads(info_path.read_text())
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.mean = np.array(self.info["mean"], np.float32)
        self.std = np.array(self.info["std"], np.float32)
        self.rotten_idx = self.info["classes"].index("rotten")

    def preprocess(self, bgr):
        """Same as eval_tf in the notebook: resize shorter side, centre crop, normalise."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        r, s = self.info["resize"], self.info["img_size"]
        scale = r / min(h, w)
        nw, nh = max(r, int(w * scale)), max(r, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (nw, nh), interpolation=interp)
        y0, x0 = int(round((nh - s) / 2)), int(round((nw - s) / 2))
        rgb = rgb[y0:y0 + s, x0:x0 + s].astype(np.float32) / 255.0
        x = (rgb - self.mean) / self.std
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def p_rotten(self, bgr_images):
        batch = np.concatenate([self.preprocess(b) for b in bgr_images])
        return self.sess.run(None, {self.input_name: batch})[0][:, self.rotten_idx]

    def verdict(self, p):
        if p >= self.info["t_rotten"]:
            return "ROTTEN"
        return "FRESH" if p < self.info["t_fresh"] else "UNSURE"


def check_images(model, paths):
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"{p}: could not read image"); continue
        prob = float(model.p_rotten([img])[0])
        print(f"{p}: {model.verdict(prob):6s}  P(rotten)={prob:.3f}")


def draw_result(view, verdict, p, n_sides):
    color, title, _ = VERDICT_STYLE[verdict]
    h, w = view.shape[:2]
    cv2.rectangle(view, (0, 0), (w, 110), color, -1)
    put_text(view, title.upper(), (20, 75), 2.2, (255, 255, 255), 5)
    put_text(view, f"worst side P(rotten) = {p:.2f}   sides checked: {n_sides}", (w - 560, 95), 0.6)


def run_camera(model, args):
    cap = open_camera(args.camera)
    sides, verdict = [], None
    print(__doc__)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped sending frames."); break
        crop, box = center_square(frame, args.roi)
        view = frame.copy()
        draw_box(view, box)
        if verdict:
            draw_result(view, verdict, max(sides), len(sides))
            if len(sides) < 3:
                put_text(view, "Turn the apple and press SPACE to check another side", (15, 145), 0.7)
        else:
            put_text(view, "Put the apple in the box and press SPACE", (15, 40))
        put_text(view, "SPACE=check  N=new apple  Q=quit", (15, view.shape[0] - 20), 0.7)
        cv2.imshow("Apple freshness checker", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n"):
            sides, verdict = [], None
        if key == ord(" "):
            crops = [crop]
            for _ in range(args.frames - 1):  # average a few frames to smooth out camera noise
                ok, f = cap.read()
                if ok:
                    crops.append(center_square(f, args.roi)[0])
            p = float(np.mean(model.p_rotten(crops)))
            sides.append(p)
            new_verdict = model.verdict(max(sides))
            print(f"side {len(sides)}: P(rotten)={p:.3f}  -> overall {new_verdict}")
            if not args.quiet and (new_verdict != verdict or len(sides) == 1):
                speak(VERDICT_STYLE[new_verdict][2])
            verdict = new_verdict
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=str(HERE / "model"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--roi", type=float, default=DEFAULT_ROI)
    ap.add_argument("--frames", type=int, default=5, help="frames averaged per check")
    ap.add_argument("--quiet", action="store_true", help="don't speak the result")
    ap.add_argument("--image", nargs="+", help="check image files instead of using the camera")
    args = ap.parse_args()

    model = AppleModel(args.model_dir)
    print(f"Loaded {model.info['model_name']}  (fresh if P(rotten) < {model.info['t_fresh']:.2f}, "
          f"rotten if >= {model.info['t_rotten']:.2f})")
    if args.image:
        check_images(model, args.image)
    else:
        run_camera(model, args)


if __name__ == "__main__":
    sys.exit(main())
