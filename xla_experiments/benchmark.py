"""Benchmark harness for comparing backends."""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import torch

from .config import BenchmarkConfig, BackendType
from .backends import ALL_BACKENDS


@dataclass
class Result:
    backend: str
    test_type: str
    config: Dict[str, Any]
    mean_ms: float
    min_ms: float
    p50_ms: float
    p99_ms: float
    peak_memory_mb: float


class Benchmark:
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[Result] = []
    
    def run_layer_benchmarks(self) -> List[Result]:
        results = []
        
        for backend_type in self.config.backends:
            backend_cls = ALL_BACKENDS.get(backend_type)
            if backend_cls is None:
                continue
            
            backend = backend_cls(self.config.device, self.config.dtype)
            if not backend.is_available():
                print(f"Skipping {backend_type.value}: not available")
                continue
            
            print(f"\n=== {backend_type.value} ===")
            
            for size in self.config.layer_sizes:
                for batch in self.config.batch_sizes:
                    # BitLinear
                    try:
                        layer = backend.get_bitlinear(size, size)
                        x = torch.randn(batch, size, device=self.config.device, dtype=self.config.dtype)
                        metrics = backend.benchmark(layer, x, self.config.warmup, self.config.iterations)
                        
                        r = Result(
                            backend=backend_type.value,
                            test_type="bitlinear",
                            config={"size": size, "batch": batch},
                            **metrics
                        )
                        results.append(r)
                        print(f"BitLinear({size}, {size}) batch={batch}: {metrics['mean_ms']:.3f}ms")
                    except Exception as e:
                        print(f"BitLinear({size}) failed: {e}")
                    
                    # HBitLinear (only power of 2 sizes)
                    if size & (size - 1) == 0:
                        try:
                            layer = backend.get_hbitlinear(size, size)
                            x = torch.randn(batch, size, device=self.config.device, dtype=self.config.dtype)
                            metrics = backend.benchmark(layer, x, self.config.warmup, self.config.iterations)
                            
                            r = Result(
                                backend=backend_type.value,
                                test_type="hbitlinear",
                                config={"size": size, "batch": batch},
                                **metrics
                            )
                            results.append(r)
                            print(f"HBitLinear({size}, {size}) batch={batch}: {metrics['mean_ms']:.3f}ms")
                        except Exception as e:
                            print(f"HBitLinear({size}) failed: {e}")
                    
                    if self.config.device == "cuda":
                        torch.cuda.empty_cache()
        
        self.results.extend(results)
        return results
    
    def run_model_benchmarks(self) -> List[Result]:
        results = []
        
        model_config = {
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_layers,
            "num_attention_heads": self.config.num_heads,
            "vocab_size": self.config.vocab_size,
            "max_position_embeddings": self.config.seq_length,
        }
        
        for backend_type in self.config.backends:
            backend_cls = ALL_BACKENDS.get(backend_type)
            if backend_cls is None:
                continue
            
            backend = backend_cls(self.config.device, self.config.dtype)
            if not backend.is_available():
                continue
            
            print(f"\n=== {backend_type.value} (Model) ===")
            
            for use_hadamard in [False, True]:
                try:
                    model = backend.get_model(model_config, use_hadamard)
                    input_ids = torch.randint(0, self.config.vocab_size, 
                                              (1, self.config.seq_length), 
                                              device=self.config.device)
                    
                    # Smaller warmup/iterations for model
                    metrics = backend.benchmark(model, input_ids, 
                                               min(3, self.config.warmup), 
                                               min(10, self.config.iterations))
                    
                    variant = "hadamard" if use_hadamard else "standard"
                    r = Result(
                        backend=backend_type.value,
                        test_type=f"model_{variant}",
                        config={"layers": self.config.num_layers, "hidden": self.config.hidden_size},
                        **metrics
                    )
                    results.append(r)
                    print(f"Model ({variant}): {metrics['mean_ms']:.3f}ms")
                except Exception as e:
                    print(f"Model (hadamard={use_hadamard}) failed: {e}")
                
                if self.config.device == "cuda":
                    torch.cuda.empty_cache()
        
        self.results.extend(results)
        return results
    
    def run_all(self) -> List[Result]:
        print("Running layer benchmarks...")
        self.run_layer_benchmarks()
        print("\nRunning model benchmarks...")
        self.run_model_benchmarks()
        return self.results
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
    
    def summary(self) -> str:
        lines = ["\n=== SUMMARY ===\n"]
        
        # Group by test type
        by_test = {}
        for r in self.results:
            key = (r.test_type, str(r.config))
            if key not in by_test:
                by_test[key] = []
            by_test[key].append(r)
        
        for (test_type, config), results in sorted(by_test.items()):
            lines.append(f"\n{test_type} {config}:")
            baseline = None
            for r in sorted(results, key=lambda x: x.mean_ms):
                if baseline is None:
                    baseline = r.mean_ms
                    speedup = "baseline"
                else:
                    speedup = f"{baseline / r.mean_ms:.2f}x"
                lines.append(f"  {r.backend:20s}: {r.mean_ms:8.3f}ms ({speedup})")
        
        return "\n".join(lines)

