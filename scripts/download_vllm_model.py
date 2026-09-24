"""Download a pinned safetensors snapshot with resume, hashes and atomic completion."""
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = "letechlead/MiMo-V2.6-Distill-Qwen-9B-INT4-W4A16-AutoRound"
REVISION = "077b1bd6bbb0d4a6f14e29c897d9ac205ca37669"
DEST = Path("D:/Models/hf") / REPO.split("/")[-1]


def download_file(url, target, size, sha=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        part = target.with_name(target.name + ".part")
        for attempt in range(5):
            offset = part.stat().st_size if part.exists() else 0
            if offset == size:
                break
            headers = {"User-Agent": "ChatBot-local-install"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                    if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                        # A host may ignore Range. Restart safely instead of appending a full file.
                        offset = 0
                        if response.status == 206:
                            raise ValueError("unexpected Content-Range")
                    with part.open("ab" if offset else "wb") as f:
                        last = 0
                        while chunk := response.read(4 * 1024 * 1024):
                            f.write(chunk)
                            offset += len(chunk)
                            if time.monotonic() - last > 10:
                                print(target.name, round(offset / size * 100, 1), "%", flush=True)
                                last = time.monotonic()
                if part.stat().st_size != size:
                    raise ValueError("incomplete download")
                break
            except Exception as exc:
                print("retry", attempt + 1, type(exc).__name__, flush=True)
                if attempt == 4:
                    raise
                time.sleep(3)
        if part.stat().st_size != size:
            raise ValueError(f"size mismatch: {target.name}")
        target = part
    if target.stat().st_size != size:
        raise ValueError(f"size mismatch: {target}")
    if sha:
        with target.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != sha:
            raise ValueError(f"SHA256 mismatch: {target.name}")
    if target.suffix == ".part":
        target.rename(target.with_suffix(""))


def main():
    api = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
    metadata = json.load(urllib.request.urlopen(api))
    entries = [f for f in metadata["siblings"] if f["rfilename"].endswith((".json", ".safetensors", ".jinja", ".md", ".txt")) or f["rfilename"] == "LICENSE"]
    DEST.mkdir(parents=True, exist_ok=True)
    for item in entries:
        name = item["rfilename"]
        target = (DEST / name).resolve()
        if not target.is_relative_to(DEST.resolve()):
            raise ValueError("path outside snapshot")
        download_file(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", target, item["size"], (item.get("lfs") or {}).get("sha256"))
        print("verified", name, flush=True)
    manifest = {"repo": REPO, "revision": REVISION, "files": entries, "verified_at": time.time()}
    (DEST / ".complete.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("COMPLETE", DEST, flush=True)


if __name__ == "__main__":
    main()
