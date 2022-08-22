import argparse
from pathlib import Path
from typing import List

from PIL import Image


def make_gif(paths: List[Path], save_path: Path) -> None:
    frames = [Image.open(path) for path in paths]
    frame_one = frames[0]
    frame_one.save(
        str(save_path),
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=100 * len(paths),
        loop=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name_format", default="policy_iter_step*")
    parser.add_argument("--save_path", default="policy_iter.gif")
    args = parser.parse_args()

    parent_dir = Path(__file__).parent
    paths = sorted(list(parent_dir.glob(args.file_name_format)))
    save_path = parent_dir / args.save_path
    make_gif(paths, save_path)
