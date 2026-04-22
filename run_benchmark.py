"""
Benchmark Mode Usage Guide
===========================
Run benchmarks with different curated datasets to validate SSM Agent
while minimizing API costs.
"""

import subprocess
import sys
from eval.curated_benchmark import estimate_cost, BENCHMARK_MODES

print("\n" + "="*70)
print("  BENCHMARK RUNNER - USAGE GUIDE")
print("="*70)

print("\n📊 AVAILABLE MODES:\n")

for mode in ["original", "mt-bench", "cab", "hybrid"]:
    info = BENCHMARK_MODES[mode]
    cost = estimate_cost(mode)
    print(f"  1️⃣  {mode.upper()}")
    print(f"     {info['name']}")
    print(f"     Size: {cost['turns']} prompts | Cost: ${cost['estimated_cost']:.2f}")
    print(f"     Savings: ${cost['savings_vs_original']:.2f} ({cost['savings_percent']:.1f}%)\n")

print("="*70)
print("\n🚀 HOW TO RUN:\n")

print("  Option 1 - Original Fish Detection Benchmark (30 turns)")
print("  $ python eval/benchmarks.py original")
print("  $ python eval/benchmarks.py")
print()

print("  Option 2 - MT-Bench Subset (10 prompts, ~70% cost savings)")
print("  $ python eval/benchmarks.py mt-bench")
print()

print("  Option 3 - CodeAssistBench Subset (8 prompts, ~73% cost savings)")
print("  $ python eval/benchmarks.py cab")
print()

print("  Option 4 - Hybrid Benchmark (15 prompts, ~67% cost savings)")
print("  $ python eval/benchmarks.py hybrid")
print()

print("="*70)
print("\n📈 UNDERSTANDING THE RESULTS:\n")

print("  Metrics Generated:")
print("  - Basic Statistics: Average, Std Dev, Median, Max, Min")
print("  - By Phase: Setup, Drift, Final performance breakdown")
print("  - Recovery Rate: How well models recover after noise")
print("  - ARR (Attribute Retention Ratio): Final/Setup constraint preservation")
print("  - Precision at Turn 30: Final exam performance")
print("  - Consistency Score: Lower variance = more stable during drift")
print("  - Win Rate: How many turns SSM beats Baseline")
print()

print("="*70)
print("\n💡 RECOMMENDATIONS:\n")

print("  For Quick Testing:")
print("  → Use 'cab' mode (8 prompts, ~15 min, $1.20)")
print()

print("  For Initial Validation:")
print("  → Use 'hybrid' mode (15 prompts, ~30 min, $2.25)")
print()

print("  For Publication-Ready Results:")
print("  → Use 'original' mode (30 prompts, ~60 min, $4.50)")
print("  → Cite: 'Fair-But-Hard Benchmark with continuous chat memory'")
print()

print("="*70)
print("\n📂 OUTPUT FILES:\n")

print("  For each run, generated in eval/results/:")
print("  - benchmark_results_{mode}.csv       (detailed scores)")
print("  - drift_analysis_graph_{mode}.png    (visualization)")
print()

print("="*70)

# Optional: Offer to run a mode
if len(sys.argv) > 1 and sys.argv[1].lower() in ["original", "mt-bench", "cab", "hybrid"]:
    mode = sys.argv[1].lower()
    print(f"\n▶️  Ready to run '{mode}' benchmark? (y/n)")
    response = input().lower()
    if response == 'y':
        print(f"\n🚀 Starting {mode} benchmark...\n")
        subprocess.run([sys.executable, "eval/benchmarks.py", mode])
