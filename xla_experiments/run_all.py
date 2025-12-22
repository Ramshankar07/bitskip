#!/usr/bin/env python3
"""Run all XLA backend experiments."""

import argparse
from .config import BenchmarkConfig, BackendType
from .benchmark import Benchmark


def main():
    parser = argparse.ArgumentParser(description="Run XLA backend benchmarks")
    parser.add_argument("--backends", nargs="+", default=None,
                        choices=[b.value for b in BackendType],
                        help="Backends to test (default: all)")
    parser.add_argument("--layer-only", action="store_true", help="Only run layer benchmarks")
    parser.add_argument("--model-only", action="store_true", help="Only run model benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (fewer iterations)")
    parser.add_argument("--output", "-o", default="benchmark_results.json", help="Output file")
    args = parser.parse_args()
    
    config = BenchmarkConfig()
    
    if args.backends:
        config.backends = [BackendType(b) for b in args.backends]
    
    if args.quick:
        config.warmup = 3
        config.iterations = 20
        config.layer_sizes = [512, 1024]
        config.batch_sizes = [1]
    
    benchmark = Benchmark(config)
    
    if args.model_only:
        benchmark.run_model_benchmarks()
    elif args.layer_only:
        benchmark.run_layer_benchmarks()
    else:
        benchmark.run_all()
    
    print(benchmark.summary())
    benchmark.save(args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

