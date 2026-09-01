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
# Records 100-124: random selection from typical waveforms
# Records 200-234: deliberately selected rare but clinically significant arrhythmias
MITBIH_RECORDS = list(range(100, 125)) + list(range(200, 235))

# Records to exclude (contain paced beats - common convention in literature)
EXCLUDED_RECORDS = [102, 104, 107]  # Known to contain paced beats

MITBIH_RECORDS_FILTERED = [r for r in MITBIH_RECORDS if r not in EXCLUDED_RECORDS]


def download_mitbih(output_dir: str = "data/raw", verbose: bool = True) -> None:
    """
    Download MIT-BIH Arrhythmia Database records from PhysioNet.

    Records are stored as .dat (signal data) and .atr (annotations) file pairs.
    Uses wfdb library to retrieve records from PhysioNet.

    Args:
        output_dir: Directory to store downloaded records.
        verbose: Print progress information.

    Raises:
        ImportError: If wfdb library is not installed.
    """
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb library required. Install with: pip install wfdb")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading MIT-BIH Arrhythmia Database to {output_dir}/")
        print(f"Total records to download: {len(MITBIH_RECORDS_FILTERED)}")
        print("Records 100-124: typical waveforms")
        print("Records 200-234: rare but clinically significant arrhythmias")
        print(
            f"Excluding {len(MITBIH_RECORDS) - len(MITBIH_RECORDS_FILTERED)} "
            f"records with paced beats"
        )

    failed = []
    for i, record_id in enumerate(MITBIH_RECORDS_FILTERED, 1):
        try:
            wfdb.rdrecord(f"mitdb/{record_id}", cache_dir=str(output_path))
            wfdb.rdann(f"mitdb/{record_id}", "atr", cache_dir=str(output_path))

            if verbose and i % 10 == 0:
                print(f"Downloaded {i}/{len(MITBIH_RECORDS_FILTERED)} records")

        except Exception as e:
            failed.append((record_id, str(e)))
            if verbose:
                print(f"Failed to download record {record_id}: {e}")

    if verbose:
        print(
            f"\nDownload complete. {len(MITBIH_RECORDS_FILTERED) - len(failed)} "
            f"records successfully downloaded."
        )
        if failed:
            print(f"Failed records: {[r[0] for r in failed]}")


def get_record_info(record_dir: str = "data/raw") -> dict:
    """
    Get information about downloaded MIT-BIH records.

    Args:
        record_dir: Directory containing downloaded records.

    Returns:
        Dictionary with dataset statistics.
    """
    if not WFDB_AVAILABLE:
        return {"error": "wfdb library not available"}

    record_path = Path(record_dir)

    total_records = 0
    total_signals = 0
    total_annotations = 0

    for record_id in MITBIH_RECORDS_FILTERED:
        try:
            record = wfdb.rdrecord(f"mitdb/{record_id}", cache_dir=str(record_path))
            ann = wfdb.rdann(f"mitdb/{record_id}", "atr", cache_dir=str(record_path))
            total_records += 1
            total_signals += record.nsig
            total_annotations += len(ann.sample)
        except Exception:
            pass

    return {
        "total_records": total_records,
        "expected_records": len(MITBIH_RECORDS_FILTERED),
        "total_signals": total_signals,
        "estimated_total_beats": total_annotations,
        "sampling_rate_hz": 360,
        "record_duration_minutes": 30,
        "resolution_bits": 11,
        "voltage_range_mv": 10,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_mitbih()
