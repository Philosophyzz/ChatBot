"""Install Live2D's original sample characters, retaining their licensing notices."""
import hashlib
import json
import urllib.request
from pathlib import Path

DEST = Path("D:/Models/Live2D")
REPO = "Live2D/CubismWebSamples"


def main():
    tree = json.load(urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/git/trees/develop?recursive=1"))
    revision = tree["sha"]
    prefix = "Samples/Resources/"
    files = [x for x in tree["tree"] if x["type"] == "blob" and x["path"].startswith((prefix + "Hiyori/", prefix + "Mao/"))]
    for item in files:
        path = DEST / item["path"].removeprefix(prefix)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(f"https://raw.githubusercontent.com/{REPO}/{revision}/{item['path']}", timeout=60).read()
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest != item["sha"]:
            raise ValueError("Git blob checksum mismatch")
        path.write_bytes(data)
        print(path.name, flush=True)
    urls = {
        "SDK-LICENSE.md": f"https://raw.githubusercontent.com/{REPO}/{revision}/LICENSE.md",
        "NOTICE.md": f"https://raw.githubusercontent.com/{REPO}/{revision}/NOTICE.md",
        "sample-model-terms.html": "https://www.live2d.com/eula/live2d-sample-model-terms_en.html",
        "free-material-license.html": "https://www.live2d.com/eula/live2d-free-material-license-agreement_en.html",
    }
    for name, url in urls.items():
        (DEST / name).write_bytes(urllib.request.urlopen(url).read())
    (DEST / "sources.json").write_text(json.dumps({"repo": REPO, "revision": revision, "files": files, "licenses": urls}, indent=2), encoding="utf-8")
    settings_path = Path(__file__).resolve().parents[1] / "data/pet_settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8-sig")) if settings_path.exists() else {}
    models = settings.setdefault("live2d_models", {})
    for key, folder in [("hiyori", "Hiyori"), ("mao", "Mao")]:
        models.setdefault(key, str(DEST / folder / (folder + ".model3.json")))
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPLETE", DEST)


if __name__ == "__main__":
    main()
