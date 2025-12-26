import argparse
import json
from pathlib import Path

def to_single_class(src_path: Path, dst_path: Path) -> None:
    d = json.loads(src_path.read_text(encoding="utf-8"))

    # Replace categories with a single "vehicle" class id=1
    d["categories"] = [
        {"id": 1, "name": "vehicle", "supercategory": "vehicle"}
    ]

    # Force all annotations into category_id=1
    anns = d.get("annotations", [])
    for a in anns:
        a["category_id"] = 1

    # Normalize image file_name to basename only (safe for different exports)
    imgs = d.get("images", [])
    for im in imgs:
        fn = im.get("file_name", "")
        fn = fn.replace("\\", "/").split("/")[-1]
        im["file_name"] = fn

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise SystemExit(f"Missing src file: {src}")

    to_single_class(src, dst)
    print(f"Wrote: {dst}")

if __name__ == "__main__":
    main()
