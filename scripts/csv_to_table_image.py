"""Turn any CSV into a report-ready table image (FT style, matching the
other exhibits).

Usage from the terminal:
    python scripts/csv_to_table_image.py results/tables/performance_metrics.csv "Performance Metrics"
    python scripts/csv_to_table_image.py results/tables/fusion_comparison.csv "Fusion Comparison" --subtitle "Discovery vs holdout" --decimals 3
"""
import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import exhibits  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to the CSV file, e.g. results/tables/performance_metrics.csv")
    parser.add_argument("title", help="Table title")
    parser.add_argument("--subtitle", default=None)
    parser.add_argument("--decimals", type=int, default=None)
    parser.add_argument("--out", default=None, help="Output PNG path (default: auto-placed in results/figures/)")
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv_path)
    out_path = pathlib.Path(args.out) if args.out else (
        exhibits.RESULTS_FIGURES_DIR / f"{csv_path.stem}_table.png")

    saved = exhibits.csv_to_table_image(csv_path, out_path, args.title,
                                         subtitle=args.subtitle, decimals=args.decimals)
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
