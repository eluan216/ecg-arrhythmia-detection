"""Download and prepare MIT-BIH Arrhythmia Database from PhysioNet."""

import logging
from pathlib import Path

try:
    import wfdb

    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

logger = logging.getLogger(__name__)

# MIT-BIH record IDs
MITBIH_RECORDS = list(range(100, 125)) + list(range(200, 235))
EXCLUDED_RECORDS = [102, 104, 107]  # paced beats — common exclusion
MITBIH_RECORDS_FILTERED = [r for r in MITBIH_RECORDS if r not in EXCLUDED_RECORDS]


def _record_files_exist(record_id: int, output_path: Path) -> bool:
    base = output_path / str(record_id)
    return base.with_suffix(".dat").exists() and base.with_suffix(".hea").exists()


def download_mitbih(output_dir: str = "data/raw", verbose: bool = True) -> None:
    """
    Download MIT-BIH records from PhysioNet into output_dir.

    Uses wfdb.dl_files (works on Colab and local; does not rely on cache_dir).
    Files are stored as {id}.dat / {id}.hea / {id}.atr under output_dir.
    """
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb library required. Install with: pip install wfdb")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading MIT-BIH Arrhythmia Database to {output_dir}/")
        print(f"Total records: {len(MITBIH_RECORDS_FILTERED)}")

    failed = []
    for i, record_id in enumerate(MITBIH_RECORDS_FILTERED, 1):
        if _record_files_exist(record_id, output_path):
            if verbose and i % 10 == 0:
                print(f"  Already present {i}/{len(MITBIH_RECORDS_FILTERED)}")
            continue
        try:
            files = [f"{record_id}.dat", f"{record_id}.hea", f"{record_id}.atr"]
            wfdb.dl_files("mitdb", dl_dir=str(output_path), files=files)
            if verbose and i % 10 == 0:
                print(f"Downloaded {i}/{len(MITBIH_RECORDS_FILTERED)} records")
        except Exception as e:
            failed.append((record_id, str(e)))
            if verbose:
                print(f"Failed to download record {record_id}: {e}")

    ok = len(MITBIH_RECORDS_FILTERED) - len(failed)
    if verbose:
        print(f"\nDownload complete. {ok} records available under {output_dir}/")
        if failed:
            print(f"Failed records: {[r[0] for r in failed]}")


def get_record_info(record_dir: str = "data/raw") -> dict:
    """Summarize downloaded MIT-BIH records on disk."""
    if not WFDB_AVAILABLE:
        return {"error": "wfdb library not available"}

    record_path = Path(record_dir)
    total_records = 0
    total_annotations = 0

    for record_id in MITBIH_RECORDS_FILTERED:
        try:
            rec_base = str(record_path / str(record_id))
            if not Path(rec_base + ".hea").exists():
                continue
            record = wfdb.rdrecord(rec_base)
            ann = wfdb.rdann(rec_base, "atr")
            total_records += 1
            total_annotations += len(ann.sample)
            _ = record.nsig
        except Exception:
            pass

    return {
        "total_records": total_records,
        "expected_records": len(MITBIH_RECORDS_FILTERED),
        "estimated_total_beats": total_annotations,
        "sampling_rate_hz": 360,
        "record_duration_minutes": 30,
        "path": str(record_path.resolve()),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_mitbih()
