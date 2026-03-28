import json
import time
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

MENDELEY_DATASET_ID = "93h9p62kg4"
MENDELEY_API_URL = f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET_ID}"


# The public API does not expose folder names, but each class in this dataset has a unique image count
# (as documented in the dataset description). We map folder_id -> class label by matching counts.
COUNT_TO_LABEL = {
    394: "leaf_curl_virus",
    307: "spider_mites",
    66: "leaf_mold",
    519: "leaf_miner",
    166: "late_blight",
    336: "insect_damage",
    103: "healthy",
    204: "early_blight",
    156: "cercospora_leaf_mold",
    376: "bacterial_spot",
    32: "other",
}


def http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
    return json.loads(raw)


def download_file(url: str, dest: Path, sleep_s: float = 0.0):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    if sleep_s:
        time.sleep(sleep_s)


def download_dataset(
    limit_per_class: int | None = None,
    polite_sleep_s: float = 0.01,
    only_labels: set[str] | None = None,
):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    meta = http_get_json(MENDELEY_API_URL)
    files = meta.get("files", [])
    if not files:
        raise SystemExit("No files found in dataset metadata. Network issue or API changed.")

    # Group by folder_id (each folder_id corresponds to a class for this dataset)
    groups: dict[str, list[dict]] = {}
    for f in files:
        folder_id = f.get("folder_id")
        if not folder_id:
            continue
        groups.setdefault(folder_id, []).append(f)

    # Map each folder_id to a label using unique counts
    folder_to_label: dict[str, str] = {}
    for folder_id, group_files in groups.items():
        n = len(group_files)
        label = COUNT_TO_LABEL.get(n)
        if label:
            folder_to_label[folder_id] = label

    unknown = [(fid, len(g)) for fid, g in groups.items() if fid not in folder_to_label]
    if unknown:
        print("WARNING: Some folder groups could not be mapped by count:")
        for fid, n in sorted(unknown, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  folder_id={fid} count={n}")

    # Download files
    downloaded = 0
    per_class_downloaded: dict[str, int] = {}
    attempted = 0

    for f in files:
        folder_id = f.get("folder_id")
        if folder_id not in folder_to_label:
            continue
        label = folder_to_label[folder_id]
        if label == "other":
            continue
        if only_labels is not None and label not in only_labels:
            continue

        if limit_per_class is not None and per_class_downloaded.get(label, 0) >= limit_per_class:
            continue

        details = f.get("content_details") or {}
        download_url = details.get("download_url")
        if not download_url:
            continue

        # Ensure unique filename to avoid collisions
        file_id = f.get("id", "file")
        original_name = f.get("filename", f"{file_id}.jpg")
        safe_name = f"{file_id}_{original_name}"
        dest = RAW_DIR / label / safe_name

        if dest.exists():
            per_class_downloaded[label] = per_class_downloaded.get(label, 0) + 1
            continue

        try:
            attempted += 1
            download_file(download_url, dest, sleep_s=polite_sleep_s)
            downloaded += 1
            per_class_downloaded[label] = per_class_downloaded.get(label, 0) + 1
            if downloaded % 50 == 0:
                print(f"Downloaded {downloaded} images so far...", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed: {label}/{safe_name} -> {exc}")

    print("Download complete.")
    print(f"Saved to: {RAW_DIR}")
    print(f"Downloaded files: {downloaded}")
    print("Per-class counts (downloaded):")
    for k in sorted(per_class_downloaded):
        print(f"  {k}: {per_class_downloaded[k]}")


if __name__ == "__main__":
    # Set limit_per_class to a small number if you want a fast demo download.
    download_dataset(limit_per_class=None)

