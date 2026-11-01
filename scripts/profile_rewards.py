"""Profile a predeclared group and every supplied runner receipt; no scoring or training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phaseeval.reward_profile import profile_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", default=[], help="repeat for every attempted episode")
    parser.add_argument("--scorecard", type=Path, action="append", default=[], help="optional hashed EpisodeScore attachment; repeat as needed")
    parser.add_argument("--out", type=Path, help="new report file; defaults to stdout")
    args = parser.parse_args()
    try:
        report = profile_group(args.manifest, args.receipt, args.scorecard)
        output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out:
            with args.out.open("x") as stream:
                stream.write(output)
        else:
            print(output, end="")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"profile error: {exc}\n")


if __name__ == "__main__":
    main()
