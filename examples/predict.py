"""Send a clearly labeled example to the running API using only the standard library."""

import argparse
import json
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", choices=["auto", "baseline", "cnn"], default="auto")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    try:
        with urllib.request.urlopen(base + "/explorer/data", timeout=10) as response:
            data = json.load(response)
        sample = data["samples"][0]
        request = urllib.request.Request(
            base + "/predict",
            data=json.dumps(
                {
                    "signal": sample["signal"],
                    "sampling_rate": data["sampling_rate"],
                    "model": args.model,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"API returned HTTP {exc.code}: {exc.read().decode()}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit("Start the API and load a trusted model bundle first.") from exc
    print(
        json.dumps(
            {
                "example_source": data["source"],
                "reference": sample.get("reference"),
                "prediction": result,
                "scope": "Research example, not a clinical decision.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
