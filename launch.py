"""Create a FiftyOne dataset from a folder of tiles and open the FiftyOne App.

The dataset is stored in the MongoDB instance that FiftyOne manages (~/.fiftyone/var/lib/mongo),
so it stays available between sessions.

The folder can be any output of tiler.py or notebook 05. Every *.tif tile in it (or in its
tiles/ subfolder) becomes one sample. If tiles_index.csv is present, its columns (licence,
producer, scene, ...) are added to the samples. FiftyOne cannot show GeoTIFFs in the browser,
so a JPEG preview is used for display: taken from the index's preview_file column when it
exists, otherwise created in <folder>/previews/.

Examples
--------
    uv run python launch.py --folder data/tiles5cm --name tiles5cm
    uv run python launch.py --folder data/tiles5cm --name tiles5cm --rebuild   # replace the dataset
    uv run python launch.py --folder data/tiles5cm --no-app                    # only create it
"""

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.warp import transform as warp_transform
import fiftyone as fo

ROOT = Path(__file__).resolve().parent
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
BASE_FIELDS = {"tile_id", "tile_file", "preview_file", "tile_path", "preview_path"}


def resolve(path_str: str) -> Path:
    """Paths in the index are relative to the project root (or absolute)."""
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def discover_tiles(folder: Path) -> pd.DataFrame:
    index_csv = folder / "tiles_index.csv"
    if index_csv.exists():
        df = pd.read_csv(index_csv)
        df["tile_path"] = [str(resolve(p)) if isinstance(p, str) and p else None
                           for p in df.get("tile_file", pd.Series([None] * len(df)))]
        # fall back to the tiles folder when the index path does not match
        missing = df.tile_path.isna() | ~df.tile_path.map(lambda p: bool(p) and Path(p).exists())
        df.loc[missing, "tile_path"] = df.loc[missing, "tile_id"].map(
            lambda t: str(folder / "tiles" / f"{t}.tif"))
    else:
        tifs = sorted(p for p in folder.rglob("*.tif") if "previews" not in p.parts)
        df = pd.DataFrame({"tile_id": [p.stem for p in tifs], "tile_path": [str(p) for p in tifs]})
    df = df[df.tile_path.map(lambda p: Path(p).exists())].reset_index(drop=True)
    if df.empty:
        sys.exit(f"No tiles found in {folder} (expected *.tif files or tiles_index.csv)")
    return df


def tile_geometry(path: str) -> dict:
    with rasterio.open(path) as src:
        west, south, east, north = src.bounds
        lon, lat = warp_transform(src.crs, "EPSG:4326", [(west + east) / 2], [(south + north) / 2])
        return {"width": src.width, "height": src.height, "gsd_m": float(abs(src.res[0])),
                "centre_lon": float(lon[0]), "centre_lat": float(lat[0])}


def make_preview(tif: str, jpg: Path) -> None:
    with rasterio.open(tif) as src:
        data = src.read([1, 2, 3])
    if data.dtype != np.uint8:
        scale = 255.0 / max(float(np.percentile(data, 99.5)), 1.0)
        data = np.clip(data * scale, 0, 255).astype(np.uint8)
    jpg.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.moveaxis(data, 0, -1)).save(jpg, quality=92)


def clean(value):
    """Convert pandas/numpy values to plain Python values FiftyOne accepts (None for NaN)."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_samples(df: pd.DataFrame, folder: Path) -> list[fo.Sample]:
    previews_dir = folder / "previews"
    samples = []
    extra_cols = [c for c in df.columns if c not in BASE_FIELDS and c != "tile_path"]
    for row in df.itertuples(index=False):
        rec = row._asdict()
        tif = rec["tile_path"]
        geo = tile_geometry(tif)

        preview = None
        if rec.get("preview_file") and isinstance(rec["preview_file"], str):
            candidate = resolve(rec["preview_file"])
            if candidate.exists():
                preview = candidate
        if preview is None:
            preview = previews_dir / f"{rec['tile_id']}.jpg"
            if not preview.exists():
                make_preview(tif, preview)

        scene = rec.get("original_id") or rec.get("scene_id") or rec["tile_id"].split("_r")[0]
        sample = fo.Sample(
            filepath=str(preview),
            tif_path=tif,
            tile_id=str(rec["tile_id"]),
            scene_id=str(scene),
            gsd_m=geo["gsd_m"],
            location=fo.GeoLocation(point=[geo["centre_lon"], geo["centre_lat"]]),
        )
        sample.metadata = fo.ImageMetadata(width=geo["width"], height=geo["height"])
        sample.tags.append(f"scene:{str(scene)[:8]}")
        for col in extra_cols:
            if col in ("original_id", "scene_id"):
                continue
            value = clean(rec.get(col))
            if value is not None and not isinstance(value, (str, int, float, bool)):
                value = str(value)
            sample[col] = value
        samples.append(sample)
    return samples


def get_or_create_dataset(folder: Path, name: str, rebuild: bool) -> fo.Dataset:
    if not NAME_PATTERN.match(name):
        sys.exit(f"Invalid dataset name {name!r}: use letters, digits, '-', '_' or '.'")
    if name in fo.list_datasets():
        if not rebuild:
            print(f"Dataset '{name}' already exists, loading it. Use --rebuild to replace it.")
            return fo.load_dataset(name)
        print(f"Deleting dataset '{name}' to rebuild it.")
        fo.delete_dataset(name)

    df = discover_tiles(folder)
    print(f"Found {len(df)} tiles in {folder}. Building samples (previews are created if missing)...")
    dataset = fo.Dataset(name, persistent=True)
    dataset.add_samples(build_samples(df, folder))
    dataset.save()
    print(f"Created '{name}' with {len(dataset)} tiles in the FiftyOne MongoDB.")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, required=True,
                        help="folder with the tiles (e.g. data/tiles5cm)")
    parser.add_argument("--name", default=None,
                        help="dataset name in FiftyOne (default: the folder name)")
    parser.add_argument("--rebuild", action="store_true", help="replace the dataset if it already exists")
    parser.add_argument("--no-app", action="store_true", help="create or load the dataset without opening the App")
    parser.add_argument("--port", type=int, default=5151, help="port for the FiftyOne App")
    args = parser.parse_args()

    folder = (ROOT / args.folder if not args.folder.is_absolute() else args.folder).resolve()
    if not folder.is_dir():
        sys.exit(f"{folder} is not a folder")
    name = args.name or folder.name

    dataset = get_or_create_dataset(folder, name, args.rebuild)
    print(dataset)

    if args.no_app:
        return
    session = fo.launch_app(dataset, port=args.port)
    print(f"FiftyOne App running at http://localhost:{args.port} (Ctrl+C to stop)")
    session.wait()


if __name__ == "__main__":
    main()
