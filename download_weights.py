"""Download experiment weights from this repository's Release and verify SHA-256."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
BASE_URL = "https://github.com/SinkCicada/Neural-Network-course-labs/releases/download/weights-v1/"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((ROOT / "weights_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest:
        target = ROOT / entry["file"]
        if target.is_file() and target.stat().st_size == entry["bytes"] and checksum(target) == entry["sha256"]:
            print("OK:", entry["file"])
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        print("Downloading:", target.name, flush=True)
        try:
            with urlopen(BASE_URL + target.name, timeout=120) as response, partial.open("wb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
            if partial.stat().st_size != entry["bytes"] or checksum(partial) != entry["sha256"]:
                raise ValueError("Weight checksum mismatch: " + target.name)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        print("Saved:", entry["file"])


if __name__ == "__main__":
    main()
