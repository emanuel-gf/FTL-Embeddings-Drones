"""Tile scene COGs into square tiles, optionally resampling to a target ground resolution.

Examples
--------
Native resolution, 640 px tiles, no overlap (default):
    uv run python tiler.py

10 cm resampling, 640 px tiles with 128 px overlap, plus a VRT per scene:
    uv run python tiler.py --gsd 0.10 --tile 640 --overlap 128 --virtual

Explicit stride instead of overlap:
    uv run python tiler.py --gsd 0.10 --tile 640 --stride 512

Outputs
-------
- Resampled scenes (only with --gsd): data/cog_<cm>cm/<scene>.tif  (Cloud-Optimized GeoTIFF)
- Tiles:  <output>/tiles/<scene>_r<row>_c<col>.tif
- Index:  <output>/tiles_index.csv  (one row per tile)
- VRT:    <output>/vrt/<scene>.vrt  (only with --virtual)

<output> defaults to data/tiling/<native|cm>_<tile>px_stride<stride>.
"""

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import rasterio
import rasterio.shutil
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window, bounds as window_bounds

ROOT = Path(__file__).resolve().parent
NATIVE_DIR = ROOT / "data" / "cog"
METADATA_CSV = ROOT / "data" / "metadata" / "images_metadata.csv"


def rel(path: Path) -> str:
    """Path relative to the project root when possible, otherwise absolute."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ground_resolution_m(src: rasterio.DatasetReader) -> tuple[float, float]:
    """Pixel size in metres (x, y). Geographic CRSs are converted at the image centre latitude."""
    res_x, res_y = abs(src.transform.a), abs(src.transform.e)
    if src.crs.is_geographic:
        west, south, east, north = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        lat = math.radians((north + south) / 2)
        res_x *= 111_320 * math.cos(lat)
        res_y *= 110_574
    return res_x, res_y


def write_cog(data, transform, crs, nodata, dst: Path) -> None:
    count, height, width = data.shape
    profile = dict(driver="GTiff", dtype=data.dtype, count=count, height=height, width=width,
                   crs=crs, transform=transform, nodata=nodata)
    with MemoryFile() as mem:
        with mem.open(**profile, tiled=True, blockxsize=512, blockysize=512) as tmp:
            tmp.write(data)
        with mem.open() as tmp:
            rasterio.shutil.copy(tmp, dst, driver="COG", COMPRESS="DEFLATE", BLOCKSIZE=512)


def resample_to_cog(src_path: Path, dst: Path, gsd: float) -> None:
    """Resample a scene to `gsd` metres per pixel (area averaging) and save it as a COG."""
    with rasterio.open(src_path) as src:
        gx, gy = ground_resolution_m(src)
        out_h = int(round(src.height * gy / gsd))
        out_w = int(round(src.width * gx / gsd))
        data = src.read(out_shape=(src.count, out_h, out_w), resampling=Resampling.average)
        transform = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
        write_cog(data, transform, src.crs, 0, dst)


def axis_starts(size: int, tile: int, stride: int) -> list[int]:
    """Start positions along one axis; the last tile is shifted so the edge is always covered."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def tile_scene(src_path: Path, tiles_dir: Path, tile: int, stride: int, min_valid: float) -> list[dict]:
    rows = []
    scene = src_path.stem
    with rasterio.open(src_path) as src:
        gsd = src.res[0]
        for r, y0 in enumerate(axis_starts(src.height, tile, stride)):
            for c, x0 in enumerate(axis_starts(src.width, tile, stride)):
                win = Window(x0, y0, tile, tile)
                data = src.read([1, 2, 3], window=win)
                valid = float((data.max(axis=0) > 0).mean())
                if valid < min_valid:
                    continue

                tile_id = f"{scene}_r{r:02d}_c{c:02d}"
                tile_file = tiles_dir / f"{tile_id}.tif"
                with rasterio.open(tile_file, "w", driver="GTiff", height=tile, width=tile, count=3,
                                   dtype=data.dtype, crs=src.crs, transform=src.window_transform(win),
                                   compress="deflate", tiled=True, blockxsize=256, blockysize=256) as out:
                    out.write(data)

                west, south, east, north = window_bounds(win, src.transform)
                lon, lat = warp_transform(src.crs, "EPSG:4326", [(west + east) / 2], [(south + north) / 2])
                rows.append({
                    "tile_id": tile_id,
                    "scene_id": scene,
                    "grid_row": r, "grid_col": c,
                    "row_px": y0, "col_px": x0,
                    "width_px": tile, "height_px": tile,
                    "gsd_m": round(gsd, 4),
                    "crs": src.crs.to_string(),
                    "valid_fraction": round(valid, 4),
                    "west": round(west, 6), "south": round(south, 6),
                    "east": round(east, 6), "north": round(north, 6),
                    "centre_lon": round(lon[0], 6), "centre_lat": round(lat[0], 6),
                    "tile_file": rel(tile_file),
                    "source_file": rel(src_path),
                })
    return rows


def build_vrt(src_path: Path, vrt_path: Path) -> None:
    gdalbuildvrt = shutil.which("gdalbuildvrt")
    if gdalbuildvrt is None:
        sys.exit("gdalbuildvrt not found. Install GDAL (brew install gdal) or use the one bundled with QGIS.")
    subprocess.run([gdalbuildvrt, "-overwrite", "-srcnodata", "0", "-vrtnodata", "0",
                    str(vrt_path), str(src_path)], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=NATIVE_DIR, help="folder with the native scene COGs")
    p.add_argument("--gsd", type=float, default=None,
                   help="resample to this ground resolution in metres (e.g. 0.10). Omit for native resolution.")
    p.add_argument("--tile", type=int, default=640, help="tile size in pixels (default 640)")
    stride_group = p.add_mutually_exclusive_group()
    stride_group.add_argument("--stride", type=int, default=None, help="step between tiles in pixels")
    stride_group.add_argument("--overlap", type=int, default=0, help="overlap between tiles in pixels (default 0)")
    p.add_argument("--min-valid", type=float, default=0.9,
                   help="skip tiles with less than this fraction of data (default 0.9)")
    p.add_argument("--virtual", action="store_true", help="also create a VRT per scene (needs GDAL)")
    p.add_argument("--output", type=Path, default=None, help="output folder for tiles and index")
    p.add_argument("--limit", type=int, default=None, help="only process the first N scenes (for tests)")
    p.add_argument("--overwrite", action="store_true", help="re-create resampled COGs that already exist")
    args = p.parse_args()

    stride = args.stride if args.stride is not None else args.tile - args.overlap
    if not 0 < stride <= args.tile:
        sys.exit(f"stride must be between 1 and the tile size ({args.tile}); got {stride}")
    if not 0 <= args.min_valid <= 1:
        sys.exit("--min-valid must be between 0 and 1")

    scenes = sorted(args.input.glob("*.tif"))
    if args.limit:
        scenes = scenes[: args.limit]
    if not scenes:
        sys.exit(f"No .tif scenes found in {args.input}")

    if args.gsd is not None:
        label = f"{round(args.gsd * 100):g}cm"
        cog_dir = ROOT / "data" / f"cog_{label}"
        cog_dir.mkdir(parents=True, exist_ok=True)
        sources = []
        for scene in scenes:
            dst = cog_dir / scene.name
            if dst.exists() and not args.overwrite:
                print(f"resampled, reusing  {rel(dst)}")
            else:
                print(f"resampling to {label}: {scene.name}")
                resample_to_cog(scene, dst, args.gsd)
            sources.append(dst)
    else:
        label = "native"
        sources = scenes

    out_dir = args.output or (ROOT / "data" / "tiling" / f"{label}_{args.tile}px_stride{stride}")
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in sources:
        scene_rows = tile_scene(src, tiles_dir, args.tile, stride, args.min_valid)
        rows.extend(scene_rows)
        print(f"{src.name}: {len(scene_rows)} tiles")

        if args.virtual:
            vrt_dir = out_dir / "vrt"
            vrt_dir.mkdir(parents=True, exist_ok=True)
            build_vrt(src, vrt_dir / f"{src.stem}.vrt")

    index = pd.DataFrame(rows)
    if METADATA_CSV.exists() and not index.empty:
        meta = pd.read_csv(METADATA_CSV)
        keep = ["item_id", "producer", "captured_start", "license", "license_type", "train_allowed"]
        keep = [c for c in keep if c in meta.columns]
        index = index.merge(meta[keep], left_on="scene_id", right_on="item_id", how="left").drop(columns="item_id")
    index.to_csv(out_dir / "tiles_index.csv", index=False)

    print(f"\nDone: {len(index)} tiles from {len(sources)} scenes -> {rel(out_dir)}/")
    print(f"Tile size {args.tile} px, stride {stride} px, resolution {label}, min valid {args.min_valid:.0%}")
    if args.virtual:
        print(f"VRT files in {rel(out_dir / 'vrt')}/")


if __name__ == "__main__":
    main()
