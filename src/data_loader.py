"""Loading the NK dataset: download, chunked CSV reading, dtype optimisation, Parquet."""
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .config import CHUNKSIZE, CSV_DTYPES, DATA_PROCESSED, DATA_RAW, HF_BASE_URL, SPLIT_FILES
from .utils import df_memory_mb


def ensure_raw_files() -> dict:
    """Return paths of train/test CSV; download them from Hugging Face if missing."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, fname in SPLIT_FILES.items():
        path = DATA_RAW / fname
        if not path.exists():
            url = f"{HF_BASE_URL}/{fname}"
            print(f"Downloading {url} ...")
            try:
                urllib.request.urlretrieve(url, path)
            except OSError as err:  # network error, HTTP error, permission ...
                raise FileNotFoundError(
                    f"{path} is missing and download failed ({err}). "
                    f"Download it manually from {url} into {DATA_RAW}."
                ) from err
        paths[split] = path
    return paths


def read_csv_chunked(path: Path, chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    """Read a long-format CSV in chunks with compact dtypes (keeps peak memory low)."""
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=CSV_DTYPES,
                             parse_dates=["timestamp"], keep_default_na=True):
        chunk["flag"] = chunk["flag"].astype("string").fillna("").astype("category")
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    # categories may differ between chunks -> unify
    for col in ["well_id", "well_type", "parameter", "unit", "flag"]:
        df[col] = df[col].astype("category")
    return df


def load_dataset() -> pd.DataFrame:
    """Load train + test into one frame with a `split` column (chronological split from the dataset)."""
    paths = ensure_raw_files()
    frames = []
    for split, path in paths.items():
        part = read_csv_chunked(path)
        part["split"] = split
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    for col in ["well_id", "well_type", "parameter", "unit", "flag", "split"]:
        df[col] = df[col].astype("category")
    return df.sort_values(["well_id", "timestamp", "parameter"], ignore_index=True)


def save_parquet(df: pd.DataFrame, name: str = "measurements_long.parquet") -> Path:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / name
    df.to_parquet(path, index=False)
    return path


def load_parquet(name: str = "measurements_long.parquet") -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / name)


def benchmark_storage() -> pd.DataFrame:
    """Compare CSV (default dtypes) vs CSV (chunked + optimised dtypes) vs Parquet:
    load time, in-memory size and file size."""
    paths = ensure_raw_files()
    train_csv = paths["train"]
    rows = []

    t0 = time.perf_counter()
    raw = pd.read_csv(train_csv)
    rows.append({"approach": "CSV, default dtypes (baseline)", "load_s": time.perf_counter() - t0,
                 "memory_mb": df_memory_mb(raw), "file_mb": train_csv.stat().st_size / 1e6})

    t0 = time.perf_counter()
    opt = read_csv_chunked(train_csv)
    rows.append({"approach": "CSV, chunksize + optimised dtypes", "load_s": time.perf_counter() - t0,
                 "memory_mb": df_memory_mb(opt), "file_mb": train_csv.stat().st_size / 1e6})

    pq = save_parquet(opt, "train_benchmark.parquet")
    t0 = time.perf_counter()
    back = pd.read_parquet(pq)
    rows.append({"approach": "Parquet (snappy)", "load_s": time.perf_counter() - t0,
                 "memory_mb": df_memory_mb(back), "file_mb": pq.stat().st_size / 1e6})

    res = pd.DataFrame(rows)
    res[["load_s", "memory_mb", "file_mb"]] = res[["load_s", "memory_mb", "file_mb"]].round(3)
    return res
