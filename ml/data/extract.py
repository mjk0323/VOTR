"""Selectively extracts a small sample of singers from the AI Hub "다음색 가이드보컬"
dataset zips, without unzipping the full ~32GB archive.

The source (원천데이터) and label (라벨링데이터) zips mirror the same folder structure
(GENDER/AGE/GENRE/TONE/SINGER_ID/...), just with .wav/.mid/.csv vs .json files - so we
pick a handful of singer folders from the label zip's file listing and pull the matching
members from both zips.

Usage:
    python extract.py --split training --singers 2
    python extract.py --split validation --singers 1
"""

import argparse
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "177.다음색 가이드보컬 데이터" / "01.데이터"
OUTPUT_ROOT = Path(__file__).resolve().parent / "extracted"

SPLITS = {
    "training": {
        "source_zip": DATASET_ROOT / "1.Training" / "원천데이터" / "TS1.zip",
        "label_zip": DATASET_ROOT / "1.Training" / "라벨링데이터" / "TL1.zip",
    },
    "validation": {
        "source_zip": DATASET_ROOT / "2.Validation" / "원천데이터" / "VS1.zip",
        "label_zip": DATASET_ROOT / "2.Validation" / "라벨링데이터" / "VL1.zip",
    },
}


def pick_singer_json_members(label_zip: Path, max_singers: int) -> list[str]:
    with zipfile.ZipFile(label_zip) as zf:
        json_names = [n for n in zf.namelist() if n.endswith(".json")]

    singer_dirs: list[str] = []
    for name in json_names:
        parent = str(Path(name).parent)
        if parent not in singer_dirs:
            singer_dirs.append(parent)
        if len(singer_dirs) >= max_singers:
            break

    selected_dirs = set(singer_dirs)
    return [n for n in json_names if str(Path(n).parent) in selected_dirs]


def extract_sample(split: str, max_singers: int) -> Path:
    paths = SPLITS[split]
    source_zip, label_zip = paths["source_zip"], paths["label_zip"]

    json_members = pick_singer_json_members(label_zip, max_singers)
    out_dir = OUTPUT_ROOT / split
    labels_out = out_dir / "labels"
    audio_out = out_dir / "audio"
    labels_out.mkdir(parents=True, exist_ok=True)
    audio_out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(label_zip) as zf:
        for name in json_members:
            zf.extract(name, labels_out)

    wav_members = [str(Path(n).with_suffix(".wav")).replace("\\", "/") for n in json_members]
    with zipfile.ZipFile(source_zip) as zf:
        available = set(zf.namelist())
        extracted = 0
        for name in wav_members:
            if name in available:
                zf.extract(name, audio_out)
                extracted += 1

    print(f"[{split}] {len(json_members)} label files, {extracted} matching wav files extracted")
    print(f"  labels -> {labels_out}")
    print(f"  audio  -> {audio_out}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["training", "validation"], required=True)
    parser.add_argument("--singers", type=int, default=2, help="number of singer folders to extract")
    args = parser.parse_args()

    extract_sample(args.split, args.singers)
