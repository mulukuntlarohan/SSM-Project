"""
SSM Framework - Evaluation Package
"""

from .benchmarks import (
    run_benchmark,
    evaluate_response,
    plot_results,
    print_summary,
    main
)

from .curated_benchmark import (
    BENCHMARK_MODES,
    estimate_cost,
    MT_BENCH_PROMPTS,
    CAB_PROMPTS,
    HYBRID_PROMPTS
)

__all__ = [
    'run_benchmark',
    'evaluate_response',
    'plot_results',
    'print_summary',
    'main',
    'BENCHMARK_MODES',
    'estimate_cost',
    'MT_BENCH_PROMPTS',
    'CAB_PROMPTS',
    'HYBRID_PROMPTS',
]
