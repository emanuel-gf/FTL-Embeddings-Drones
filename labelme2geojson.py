"""Convert labelme annotations (JSON) into standard GeoJSON files, one per label.

Each labelme JSON refers to a tile (imagePath). The tile is a georeferenced GeoTIFF,
so pixel coordinates are converted to map coordinates with its transform and then
to WGS84 longitude/latitude, as GeoJSON (RFC 7946) requires.

GeoJSON cannot hold several layers, so every label becomes its own file:
    <input name>_<label>.geojson    e.g. 68a4a953e50784296730bf0f_1_building.geojson

Examples
--------
One JSON file, written next to it:
    uv run python labelme2geojson.py data/labelme/images/68a4a953e50784296730bf0f_1.json

Choose the output folder (names still follow the input):
    uv run python labelme2geojson.py data/labelme/images/68a4a953e50784296730bf0f_1.json --output data/labelme/geojson/

A whole folder (all JSON files are merged, one file per label):
    uv run python labelme2geojson.py data/labelme/images --output data/labelme/geojson/

Also write a GeoPackage with one layer per label (choose the layers when importing in QGIS):
    uv run python labelme2geojson.py data/labelme/images --output data/labelme/geojson/ --gpkg

Each output feature keeps its label, shape type, description, group id, the tile and
its original scene, and the licence information from data/labelme/labelme_tiles.csv.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from shapely.geometry import LineString, MultiPoint, Point, Polygon, mapping
from shapely.validation import make_valid

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = ROOT / "data" / "labelme" / "images"
TILES_CSV = ROOT / "data" / "labelme" / "labelme_tiles.csv"
CIRCLE_SEGMENTS = 64

SUPPORTED = {"polygon", "rectangle", "oriented_rectangle", "circle", "point", "points", "line", "linestrip"}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def pixel_to_geometry(shape_type: str, pts: list[list[float]]):
    """Return a shapely geometry in pixel coordinates (x = column, y = row)."""
    xy = [(float(x), float(y)) for x, y in pts]
    if shape_type == "polygon":
        if len(xy) < 3:
            raise ValueError("polygon needs at least 3 points")
        return Polygon(xy)
    if shape_type in ("rectangle", "oriented_rectangle"):
        if shape_type == "rectangle":
            (x0, y0), (x1, y1) = xy[0], xy[1]
            return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        return Polygon(xy)  # oriented rectangle: 4 corners in order
    if shape_type == "circle":
        (cx, cy), (ex, ey) = xy[0], xy[1]
        r = math.hypot(ex - cx, ey - cy)
        ring = [(cx + r * math.cos(2 * math.pi * i / CIRCLE_SEGMENTS),
                 cy + r * math.sin(2 * math.pi * i / CIRCLE_SEGMENTS)) for i in range(CIRCLE_SEGMENTS)]
        return Polygon(ring)
    if shape_type == "point":
        return Point(xy[0])
    if shape_type == "points":
        return MultiPoint(xy)
    if shape_type in ("line", "linestrip"):
        if len(xy) < 2:
            raise ValueError(f"{shape_type} needs at least 2 points")
        return LineString(xy)
    raise ValueError(f"unsupported shape_type {shape_type!r}")


def to_wgs84(geom, transform: Affine, crs) -> dict:
    """Map pixel coordinates -> map CRS -> WGS84 lon/lat, returning GeoJSON geometry."""
    def convert(coords):
        xs, ys = zip(*coords)
        mx, my = [], []
        for x, y in zip(xs, ys):
            mx_, my_ = transform * (x, y)
            mx.append(mx_)
            my.append(my_)
        lon, lat = warp_transform(crs, "EPSG:4326", mx, my)
        return [[round(a, 8), round(b, 8)] for a, b in zip(lon, lat)]

    if geom.geom_type == "Polygon":
        ring = convert(list(geom.exterior.coords))
        if not Polygon(ring).is_valid:
            fixed = make_valid(Polygon(ring))
            return mapping(fixed)
        return {"type": "Polygon", "coordinates": [ring]}
    if geom.geom_type == "Point":
        return {"type": "Point", "coordinates": convert([geom.coords[0]])[0]}
    if geom.geom_type == "MultiPoint":
        return {"type": "MultiPoint", "coordinates": convert([g.coords[0] for g in geom.geoms])}
    if geom.geom_type == "LineString":
        return {"type": "LineString", "coordinates": convert(list(geom.coords))}
    raise ValueError(f"cannot convert geometry {geom.geom_type}")


def find_tile(json_path: Path, image_path: str, images_dir: Path) -> Path:
    for candidate in (json_path.parent / image_path, images_dir / Path(image_path).name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"tile {image_path!r} not found next to {json_path.name} or in {images_dir}")


def load_tile_info() -> dict:
    if not TILES_CSV.exists():
        return {}
    df = pd.read_csv(TILES_CSV)
    df["file_name"] = df.tile_file.map(lambda p: Path(p).name)
    keep = [c for c in ["original_id", "tile_id", "license", "attribution", "producer", "captured_start"] if c in df.columns]
    return {row["file_name"]: {c: row[c] for c in keep} for _, row in df.iterrows()}


def convert_file(json_path: Path, images_dir: Path, tile_info: dict, stats: dict) -> list[dict]:
    data = json.loads(json_path.read_text())
    if "shapes" not in data:
        raise ValueError(f"{json_path.name} is not a labelme annotation (no 'shapes')")

    tile_path = find_tile(json_path, data["imagePath"], images_dir)
    features = []
    with rasterio.open(tile_path) as src:
        if src.crs is None:
            raise ValueError(f"{tile_path.name} has no CRS, cannot georeference the labels")
        for shape in data["shapes"]:
            shape_type = shape.get("shape_type", "polygon")
            if shape_type not in SUPPORTED:
                stats["skipped"][shape_type] = stats["skipped"].get(shape_type, 0) + 1
                continue
            geom = pixel_to_geometry(shape_type, shape["points"])
            props = {
                "label": shape.get("label"),
                "shape_type": shape_type,
                "description": shape.get("description") or None,
                "group_id": shape.get("group_id"),
                "source_json": json_path.name,
                "tile_file": tile_path.name,
            }
            props.update(tile_info.get(tile_path.name, {}))
            features.append({
                "type": "Feature",
                "geometry": to_wgs84(geom, src.transform, src.crs),
                "properties": props,
            })
    return features


def collect_jsons(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    jsons = sorted(p for p in path.glob("*.json"))
    if not jsons:
        sys.exit(f"No .json files found in {path}")
    return jsons


def safe_name(label) -> str:
    """File-name safe version of a label (keeps letters, digits, '-' and '_')."""
    text = str(label) if label else "unlabeled"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "unlabeled"


def convert(path: Path, output: Path | None = None, images_dir: Path | None = None,
            gpkg: bool = False) -> list[Path]:
    """Convert labelme JSON(s) into one GeoJSON per label (and optionally one GeoPackage).

    Output names follow the input name: <input>_<label>.geojson. If output is a folder,
    the files go there; if output ends in .geojson, its stem is used as the prefix.
    """
    path = Path(path).resolve()
    images_dir = (images_dir or DEFAULT_IMAGES_DIR).resolve()
    base = path.stem if path.is_file() else path.name

    if output is None:
        out_dir, prefix = path.parent, base
    elif output.suffix.lower() == ".geojson":
        out_dir, prefix = output.parent, output.stem
    else:
        out_dir, prefix = output, base

    tile_info = load_tile_info()
    stats = {"skipped": {}, "files": 0, "failed": []}
    features = []
    for json_path in collect_jsons(path):
        try:
            features.extend(convert_file(json_path, images_dir, tile_info, stats))
            stats["files"] += 1
        except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as err:
            stats["failed"].append(f"{json_path.name}: {err}")

    groups: dict[str, list[dict]] = {}
    for feat in features:
        groups.setdefault(feat["properties"]["label"] or "unlabeled", []).append(feat)

    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, feats in sorted(groups.items()):
        target = out_dir / f"{prefix}_{safe_name(label)}.geojson"
        collection = {"type": "FeatureCollection", "features": feats}
        target.write_text(json.dumps(collection, ensure_ascii=False))
        written.append(target)

    if gpkg and groups:
        import geopandas as gpd

        target = out_dir / f"{prefix}.gpkg"
        if target.exists():
            target.unlink()
        for label, feats in sorted(groups.items()):
            gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
            gdf.to_file(target, layer=safe_name(label), driver="GPKG")
        written.append(target)

    print(f"Converted {len(features)} shapes from {stats['files']} file(s) into {len(groups)} label(s):")
    for label, feats in sorted(groups.items()):
        print(f"  {label}: {len(feats)} shapes")
    for target in written:
        print(f"  -> {rel(target)}")
    if not features:
        print("No supported shapes found, nothing written.")
    if stats["skipped"]:
        print(f"Skipped unsupported shape types: {stats['skipped']}")
    for msg in stats["failed"]:
        print(f"Failed: {msg}")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="a labelme .json file or a folder containing .json files")
    p.add_argument("--output", type=Path, default=None,
                   help="output folder (file named after the input); or a full path ending in .geojson")
    p.add_argument("--gpkg", action="store_true",
                   help="also write one GeoPackage with a layer per label (selectable when importing in QGIS)")
    p.add_argument("--images", type=Path, default=None,
                   help=f"folder with the tiles, used if imagePath is not found (default {rel(DEFAULT_IMAGES_DIR)})")
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"{args.input} does not exist")
    convert(args.input, args.output, args.images, gpkg=args.gpkg)


if __name__ == "__main__":
    main()
