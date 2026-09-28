"""Collect your own training photos inside the chamber.

Put an apple in the chamber, then:
  F  = save this view as FRESH        R  = save this view as ROTTEN
  N  = next apple (new apple number)  Q  = quit
Turn the apple between shots so you get 6-8 different sides of every apple.

Photos go to apple_photos/fresh/apple007_003.jpg etc. Upload the whole apple_photos
folder to Google Drive as MyDrive/apple_photos and re-run the training notebook.

If you photograph the same apple again days later (after it has gone bad), start with
  python capture_dataset.py --apple 7
so it keeps its number. That keeps every photo of one apple in the same train/val/test split.
"""
import argparse
import re
import time
from pathlib import Path

import cv2

from common import DEFAULT_ROI, center_square, draw_box, open_camera, put_text

CLASSES = ("fresh", "rotten")


def next_apple_id(out):
    ids = [int(m.group(1)) for p in out.rglob("apple*_*.jpg") if (m := re.match(r"apple(\d+)_", p.name))]
    return max(ids, default=0) + 1


def count_for(out, apple_id):
    return {c: len(list((out / c).glob(f"apple{apple_id:03d}_*.jpg"))) for c in CLASSES}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="apple_photos")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--roi", type=float, default=DEFAULT_ROI, help="size of the square crop (0-1)")
    ap.add_argument("--apple", type=int, help="apple number to start at (default: next unused)")
    ap.add_argument("--size", type=int, default=512, help="saved image size in pixels")
    args = ap.parse_args()

    out = Path(args.out)
    for c in CLASSES:
        (out / c).mkdir(parents=True, exist_ok=True)
    apple_id = args.apple or next_apple_id(out)
    cap = open_camera(args.camera)
    flash_until, flash_color = 0, (255, 255, 255)
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped sending frames."); break
        crop, box = center_square(frame, args.roi)
        view = frame.copy()
        draw_box(view, box, flash_color if time.time() < flash_until else (255, 255, 255))
        n = count_for(out, apple_id)
        totals = {c: len(list((out / c).glob("*.jpg"))) for c in CLASSES}
        put_text(view, f"Apple #{apple_id:03d}   this apple: fresh {n['fresh']}  rotten {n['rotten']}", (15, 35))
        put_text(view, f"Total: fresh {totals['fresh']}  rotten {totals['rotten']}", (15, 70), 0.7)
        put_text(view, "F=fresh  R=rotten  N=next apple  Q=quit", (15, view.shape[0] - 20), 0.7)
        cv2.imshow("Capture apple photos", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n"):
            apple_id = next_apple_id(out) if sum(n.values()) else apple_id
            print(f"Now photographing apple #{apple_id:03d}")
        if key in (ord("f"), ord("r")):
            cls = "fresh" if key == ord("f") else "rotten"
            idx = n[cls] + 1
            while (path := out / cls / f"apple{apple_id:03d}_{idx:03d}.jpg").exists():
                idx += 1
            img = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            flash_until, flash_color = time.time() + 0.3, (0, 200, 0) if cls == "fresh" else (0, 0, 255)
            print("saved", path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
