"""Build the fixed U-CUDA MNIST pilot CSV from the four CVDF IDX archives."""

import argparse
import csv
import gzip
import hashlib
import struct
from pathlib import Path


SHA256 = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}
PIXELS = 28 * 28


def archive(directory, name):
    compressed = (directory / name).read_bytes()
    if hashlib.sha256(compressed).hexdigest() != SHA256[name]:
        raise ValueError(f"SHA-256 mismatch: {name}")
    return gzip.decompress(compressed)


def records(directory, prefix, count):
    images = archive(directory, f"{prefix}-images-idx3-ubyte.gz")
    labels = archive(directory, f"{prefix}-labels-idx1-ubyte.gz")
    if struct.unpack_from(">4I", images) != (2051, count, 28, 28):
        raise ValueError(f"invalid {prefix} image header")
    if struct.unpack_from(">2I", labels) != (2049, count):
        raise ValueError(f"invalid {prefix} label header")
    if len(images) != 16 + count * PIXELS or len(labels) != 8 + count:
        raise ValueError(f"invalid {prefix} record lengths")
    for index, label in enumerate(labels[8:]):
        if label > 9:
            raise ValueError(f"invalid digit at {prefix}-{index}")
        start = 16 + index * PIXELS
        yield index, label, images[start:start + PIXELS]


def select(directory, prefix, count, splits):
    seen = [0] * 10
    limit = splits[-1][1]
    for index, label, pixels in records(directory, prefix, count):
        if seen[label] == limit:
            continue
        seen[label] += 1
        split = next(name for name, stop in splits if seen[label] <= stop)
        yield (f"{prefix}-{index}", split, label, "", *(format(p / 255, ".8g") for p in pixels))
        if all(n == limit for n in seen):
            return
    raise ValueError(f"{prefix} does not contain enough examples of every digit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("idx_directory", type=Path)
    parser.add_argument("output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--full", action="store_true", help="use all official examples, with the last 5000 training rows for validation")
    modes.add_argument("--development", action="store_true",
                       help="use only the official training archive, split 50000/5000/5000")
    args = parser.parse_args()
    if args.development:
        train = (
            (f"train-{i}", "train" if i < 50000 else "validation" if i < 55000 else "test", label, "",
             *(format(p / 255, ".8g") for p in pixels))
            for i, label, pixels in records(args.idx_directory, "train", 60000)
        )
        test = ()
    elif args.full:
        train = (
            (f"train-{i}", "train" if i < 55000 else "validation", label, "",
             *(format(p / 255, ".8g") for p in pixels))
            for i, label, pixels in records(args.idx_directory, "train", 60000)
        )
        test = (
            (f"t10k-{i}", "test", label, "", *(format(p / 255, ".8g") for p in pixels))
            for i, label, pixels in records(args.idx_directory, "t10k", 10000)
        )
    else:
        train = select(args.idx_directory, "train", 60000, (("train", 20), ("validation", 25)))
        test = select(args.idx_directory, "t10k", 10000, (("test", 5),))
    with args.output.open("x", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("id", "split", "label", "group", *(f"pixel{i}" for i in range(PIXELS))))
        writer.writerows(train)
        writer.writerows(test)


if __name__ == "__main__":
    main()
