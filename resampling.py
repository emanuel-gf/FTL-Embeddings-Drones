"""Change the ground resolution of the GeoTIFF scenes and save them as Cloud-Optimized GeoTIFFs (COG).

Only rasterio functions are used (rasterio.open, calculate_default_transform, reproject,
rasterio.shutil.copy). By default each scene keeps its own CRS; pass --crs only if you need another one.

Examples
--------
10 cm, same CRS (default), from data/cog into data/cog_10cm:
    uv run python resampling.py --gsd 0.10

Same, with an explicit input and output folder:
    uv run python resampling.py --input data/cog --output data/cog_10cm --gsd 0.10

Scenes that already exist in the output folder are skipped (use --overwrite to redo them).
"""

import argparse
import sys
from pathlib import Path

import rasterio
import rasterio.shutil
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "cog", help="folder with the native scenes")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "cog_10cm", help="folder for the resampled COGs")
    parser.add_argument("--gsd", type=float, default=0.10, help="ground sampling distance in metres per pixel")
    parser.add_argument("--crs", default=None,
                        help="optional target CRS, e.g. EPSG:31984. Default: keep the CRS of each scene")
    parser.add_argument("--overwrite", action="store_true", help="redo the scenes that already exist")
    args = parser.parse_args()

    scenes = sorted(args.input.glob("*.tif"))
    if not scenes:
        sys.exit(f"No .tif scenes found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    for path in scenes:
        dst = args.output / path.name
        if dst.exists() and not args.overwrite:
            print(f"skipped (already exists): {dst}")
            continue

        with rasterio.open(path) as src:
            dst_crs = CRS.from_string(args.crs) if args.crs else src.crs
            if dst_crs.is_geographic:
                sys.exit(f"{path.name} is in a geographic CRS (degrees): pass a metric CRS, "
                         f"for example --crs EPSG:31984")

            # Output grid: same extent as the scene, pixels of `gsd` metres
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=args.gsd)
            profile = dict(driver="GTiff", dtype=src.dtypes[0], count=src.count, width=width, height=height,
                           crs=dst_crs, transform=transform, nodata=src.nodata)

            with MemoryFile() as mem:
                with mem.open(**profile) as tmp:
                    for band in range(1, src.count + 1):
                        reproject(source=rasterio.band(src, band), destination=rasterio.band(tmp, band),
                                  src_transform=src.transform, src_crs=src.crs,
                                  dst_transform=transform, dst_crs=dst_crs,
                                  resampling=Resampling.average)
                with mem.open() as tmp:
                    rasterio.shutil.copy(tmp, str(dst), driver="COG", COMPRESS="DEFLATE", BLOCKSIZE=512)

        print(f"resampled to {args.gsd * 100:g} cm: {dst}  ({width} x {height} px, {dst_crs.to_string()})")

    print(f"Done: {len(scenes)} scene(s) checked in {args.output}")


if __name__ == "__main__":
    main()
