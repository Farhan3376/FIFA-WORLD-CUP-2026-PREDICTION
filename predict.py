#!/usr/bin/env python3
"""FIFA World Cup 2026 Match Winner Prediction CLI.

This script accepts a home team and an away team and returns the predicted
match outcome, including probability distribution and confidence score,
in JSON format.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to python path to ensure imports work
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.inference import InferenceWrapper


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict the outcome of a match between two international football teams."
    )
    parser.add_argument(
        "--home",
        required=True,
        help="Name of the home team (e.g., 'Argentina', 'France')."
    )
    parser.add_argument(
        "--away",
        required=True,
        help="Name of the away team (e.g., 'Brazil', 'Germany')."
    )
    parser.add_argument(
        "--neutral",
        action="store_true",
        help="Flag indicating if the match is played on a neutral venue (default: False)."
    )
    parser.add_argument(
        "--tournament-importance",
        type=float,
        default=1.0,
        help="Weight/importance of the tournament (default: 1.0, e.g., World Cup)."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Year of the predicted match (default: 2026)."
    )
    parser.add_argument(
        "--month",
        type=int,
        default=6,
        help="Month of the predicted match (default: 6)."
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print the output JSON with indentations."
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_arguments()
    
    try:
        # Initialize the inference wrapper
        wrapper = InferenceWrapper()
        
        # Predict the match outcome
        prediction = wrapper.predict_match(
            home_team_name=args.home,
            away_team_name=args.away,
            is_neutral=args.neutral,
            tournament_importance=args.tournament_importance,
            year=args.year,
            month=args.month
        )
        
        # Output JSON result
        indent = 4 if args.pretty else None
        print(json.dumps(prediction, indent=indent))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "status": "failed"
        }
        print(json.dumps(error_result), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
