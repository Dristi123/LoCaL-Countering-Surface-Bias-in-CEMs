
import csv
from collections import defaultdict

CSV_FILE = "metrics_usage_filtered.csv"

counts = defaultdict(lambda: defaultdict(int))

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        metric = row["metric"].strip()
        year = str(row.get("year", "")).strip() or "Unknown"
        counts[metric][year] += 1


print("========== Metric Usage Report ==========")
grand_total = 0
for metric, year_counts in counts.items():
    total = sum(year_counts.values())
    grand_total += total
    print(f"▶ {metric}: {total} mentions")
    for year, c in sorted(year_counts.items()):
        print(f"   {year}: {c}")
    print()

print(f"=== Overall total: {grand_total} mentions ===")
