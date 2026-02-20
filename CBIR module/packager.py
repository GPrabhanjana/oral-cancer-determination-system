"""
========================================================
  CBIR Index Builder — Oral Cancer Lesion Dataset
========================================================

Scans the dataset folder, extracts MobileNetV2 embeddings
for every image, and saves the index to cbir_index.npz.

USAGE:
  python build_index.py
  python build_index.py --dataset_dir "./Oral Images Dataset"
  python build_index.py --dataset_dir "./Oral Images Dataset" --device cuda

DATASET FOLDER STRUCTURE EXPECTED (auto-detected recursively):
  Oral Images Dataset/
    original_data/
      benign_lesions/   OR  benign/
      malignant_lesions/  OR  malignant/
    augmented_data/
      augmented_benign/
      augmented_malignant/
"""

import os
import sys
import ssl
import argparse
import numpy as np
from pathlib import Path
import urllib.request as _urllib_request

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
    from PIL import Image
except ImportError as e:
    print(f"\n[!] Missing dependency: {e}")
    print("    Run:  pip install torch torchvision pillow\n")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
INDEX_PATH     = "cbir_index.npz"
IMG_SIZE       = 224
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

FOLDER_LABEL_MAP = {
    "benign_lesions":      "benign",
    "malignant_lesions":   "malignant",
    "augmented_benign":    "benign",
    "augmented_malignant": "malignant",
    "benign":              "benign",
    "malignant":           "malignant",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SSL helpers
# ══════════════════════════════════════════════════════════════════════════════
def _disable_ssl_verification():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = _urllib_request.build_opener(_urllib_request.HTTPSHandler(context=ctx))
    _urllib_request.install_opener(opener)

def _restore_ssl():
    _urllib_request.install_opener(_urllib_request.build_opener())


# ══════════════════════════════════════════════════════════════════════════════
#  Feature Extractor
# ══════════════════════════════════════════════════════════════════════════════
class MobileNetV2Extractor:
    """MobileNetV2 → 1280-d L2-normalised embedding."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        print("  Loading MobileNetV2 weights (may download ~14 MB on first run)...")
        try:
            backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        except Exception as e:
            if "CERTIFICATE" in str(e).upper() or "SSL" in str(e).upper():
                print("  [!] SSL error — retrying with verification disabled...")
                _disable_ssl_verification()
                try:
                    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                finally:
                    _restore_ssl()
            else:
                raise

        self.model = nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        emb = self.model(tensor).squeeze().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset discovery
# ══════════════════════════════════════════════════════════════════════════════
def discover_image_folders(dataset_dir: Path) -> list[tuple[Path, str]]:
    found = []
    for dirpath, dirnames, _ in os.walk(dataset_dir):
        folder = Path(dirpath)
        label  = FOLDER_LABEL_MAP.get(folder.name.lower())
        if label:
            found.append((folder, label))
            dirnames.clear()
    return found


# ══════════════════════════════════════════════════════════════════════════════
#  Index builder
# ══════════════════════════════════════════════════════════════════════════════
def build_index(dataset_dir: str, extractor: MobileNetV2Extractor) -> None:
    dataset_dir = Path(dataset_dir)
    embeddings, paths, labels, sources = [], [], [], []
    total, errors = 0, 0

    folders = discover_image_folders(dataset_dir)
    if not folders:
        raise RuntimeError(
            f"No recognised folders found under '{dataset_dir}'.\n"
            f"Expected folder names: {list(FOLDER_LABEL_MAP.keys())}"
        )

    print(f"\n  Found {len(folders)} image folder(s):\n")
    for folder, label in sorted(folders, key=lambda x: str(x[0])):
        image_files = [f for f in folder.iterdir()
                       if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
        rel = folder.relative_to(dataset_dir)
        print(f"  [{label.upper():<10}]  {len(image_files):>5} images  ←  {rel}")

        for img_path in image_files:
            try:
                img = Image.open(img_path)
                emb = extractor.extract(img)
                embeddings.append(emb)
                paths.append(str(img_path))
                labels.append(label)
                sources.append(folder.name)
                total += 1
                if total % 100 == 0:
                    print(f"    ... {total} images processed")
            except Exception as e:
                errors += 1
                print(f"    [!] Skipping {img_path.name}: {e}")

    if total == 0:
        raise RuntimeError("No images were successfully processed.")

    np.savez_compressed(
        INDEX_PATH,
        embeddings = np.array(embeddings, dtype=np.float32),
        paths      = np.array(paths,      dtype=object),
        labels     = np.array(labels,     dtype=object),
        sources    = np.array(sources,    dtype=object),
    )

    b = labels.count("benign")
    m = labels.count("malignant")
    print(f"\n  ✓ Index built: {total} total  ({b} benign, {m} malignant)")
    print(f"  ✓ Saved → {INDEX_PATH}")
    if errors:
        print(f"  [!] {errors} images skipped.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Build CBIR index from oral cancer image dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="./Oral Images Dataset",
        help='Path to dataset root (default: "./Oral Images Dataset").',
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
        help="Compute device (default: cpu).",
    )
    parser.add_argument(
        "--download_weights", action="store_true",
        help="Download MobileNetV2 weights (handles SSL) then exit.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    print(f"\n{'─'*50}")
    print(f"  Oral Cancer CBIR — Index Builder")
    print(f"  device={args.device}  output={INDEX_PATH}")
    print(f"{'─'*50}")

    if args.download_weights:
        print("\n  Downloading MobileNetV2 weights...")
        _disable_ssl_verification()
        try:
            models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            print("  ✓ Done.")
        finally:
            _restore_ssl()
        sys.exit(0)

    print("\n  Loading MobileNetV2 feature extractor...")
    extractor = MobileNetV2Extractor(device=args.device)
    print("  ✓ Ready\n")

    print(f"  Scanning dataset: {args.dataset_dir}")
    build_index(args.dataset_dir, extractor)
    print("\n  Done! Run run_app.py to launch the search interface.\n")


if __name__ == "__main__":
    main()