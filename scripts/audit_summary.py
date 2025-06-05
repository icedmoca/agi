#!/usr/bin/env python3
import argparse
from datetime import datetime
from typing import Any
from core.audit import AuditLogger

def format_metric(key: str, value: Any) -> str:
    """Format a metric for display"""
    if isinstance(value, float):
        if key == "success_rate":
            return f"{value}%"
        return f"{value:.2f}"
    return str(value)

def main():
    parser = argparse.ArgumentParser(description="Display AGI system audit summary")
    parser.add_argument("--hours", type=float, help="Time window in hours")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()

    audit = AuditLogger()
    summary = audit.log_summary(window_hours=args.hours)

    if not summary:
        print("No audit log found.")
        return

    if args.json:
        import json
        print(json.dumps(summary, indent=2))
        return

    # Pretty print summary
    print("\n📊 Audit Summary")
    print("=" * 40)
    
    if args.hours:
        print(f"Window: Last {args.hours} hours")
    
    metrics = [
        ("Total Tasks", summary["total_tasks"]),
        ("Unique Goals", summary["unique_goals"]),
        ("Successes", summary["successes"]),
        ("Failures", summary["failures"]),
        ("Retries", summary["retries"]),
        ("Success Rate", f"{summary['success_rate']}%"),
        ("Average Score", summary["avg_score"]),
        ("Total Score", summary["total_score"])
    ]

    for label, value in metrics:
        if value is not None:
            print(f"{label:>15}: {value}")

    print("=" * 40)

if __name__ == "__main__":
    main() 