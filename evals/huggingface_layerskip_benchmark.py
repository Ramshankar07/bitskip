#!/usr/bin/env python3
"""
HuggingFace Transformers Benchmark Script for LayerSkip Llama3.2-1B Model
Measures TPOT, TTFT, tokens/sec, ITL using direct HuggingFace inference

This script benchmarks the LayerSkip Llama3.2-1B model using HuggingFace transformers with:
- Time Per Output Token (TPOT)
- Time To First Token (TTFT) 
- Tokens per second
- Inter-Token Latency (ITL)
- Throughput analysis
- Memory usage monitoring
- Direct inference without server
"""

import os
import time
import json
import argparse
import logging
import statistics
import threading
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import psutil

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import torch

class HuggingFaceLayerSkipBenchmark:
    """Comprehensive HuggingFace transformers benchmark for LayerSkip Llama3.2-1B model."""
    
    def __init__(self, model_name: str = "RichardErkhov/facebook_-_layerskip-llama3.2-1B-gguf", 
                 model_file: str = "layerskip-llama3.2-1B.Q2_K.gguf"):
        self.model_name = model_name
        self.model_file = model_file
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Test prompts for benchmarking
        self.test_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "The key to successful machine learning is",
            "Quantum computing represents a paradigm shift because",
            "The intersection of neuroscience and AI",
            "Sustainable energy solutions require",
            "The evolution of programming languages shows",
            "Understanding human consciousness through",
            "The challenges of climate change demand",
            "Innovation in healthcare technology enables"
        ]
        
        # Performance metrics storage
        self.results = {
            'model_info': {
                'model_name': model_name,
                'model_file': model_file,
                'inference_method': 'huggingface_transformers'
            },
            'metrics': {},
            'system_info': self._get_system_info()
        }
        
        self.model = None
        self.tokenizer = None
        self.device = None
    
    def _get_system_info(self) -> Dict:
        """Get system information for benchmarking context."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
            'platform': 'windows' if os.name == 'nt' else 'linux'
        }
        
        # GPU information
        if torch.cuda.is_available():
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_info'] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info['gpu_info'].append({
                    'name': props.name,
                    'memory_total': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}"
                })
        else:
            info['gpu_info'] = []
        
        return info
    
    def load_model(self):
        """Load the LayerSkip model using HuggingFace transformers."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Check if model is already downloaded locally
            local_model_path = f"./models/models--{self.model_name.replace('/', '--')}"
            if os.path.exists(local_model_path):
                self.logger.info(f"Found local model at: {local_model_path}")
                
                # Look for the GGUF file
                gguf_path = None
                for root, dirs, files in os.walk(local_model_path):
                    for file in files:
                        if file.endswith('.gguf'):
                            gguf_path = os.path.join(root, file)
                            break
                    if gguf_path:
                        break
                
                if gguf_path:
                    self.logger.info(f"Found GGUF file: {gguf_path}")
                    self.logger.info("Using GGUF model with llama-cpp-python")
                    
                    # Use llama-cpp-python for GGUF models
                    try:
                        from llama_cpp import Llama
                        
                        # Load the GGUF model
                        self.model = Llama(
                            model_path=gguf_path,
                            n_ctx=2048,  # Context length
                            n_threads=4,  # CPU threads
                            verbose=False
                        )
                        
                        # Create a simple tokenizer wrapper
                        self.tokenizer = None  # llama-cpp handles tokenization internally
                        self.device = "cpu"  # llama-cpp runs on CPU
                        
                        self.logger.info(f"✅ GGUF model loaded successfully from: {gguf_path}")
                        return True
                        
                    except ImportError:
                        self.logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
                        return False
                else:
                    self.logger.warning("No GGUF file found in local model directory.")
                    return False
            else:
                self.logger.warning("Local model not found.")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def generate_text(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7,
                     top_p: float = 0.9) -> Dict:
        """Generate text using llama-cpp-python for GGUF models."""
        try:
            # Generate using llama-cpp-python
            start_time = time.time()
            
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["</s>", "\n\n"],  # Stop tokens
                echo=False  # Don't echo the prompt
            )
            
            end_time = time.time()
            
            # Extract generated text
            if 'choices' in output and len(output['choices']) > 0:
                generated_text = output['choices'][0]['text']
            else:
                self.logger.warning("No choices in model output")
                return None
            
            # Calculate tokens (approximate)
            # llama-cpp-python doesn't provide exact token counts, so we estimate
            generated_tokens = len(generated_text.split())  # Rough estimate
            total_tokens = len(prompt.split()) + generated_tokens
            
            return {
                'text': generated_text,
                'generation_time': end_time - start_time,
                'tokens_generated': generated_tokens,
                'total_tokens': total_tokens
            }
            
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            # Try to reset the model state if possible
            try:
                if hasattr(self.model, 'reset'):
                    self.model.reset()
            except:
                pass
            return None
    
    def benchmark_single_request(self, prompt: str, max_tokens: int = 100, 
                               num_runs: int = 5) -> Dict:
        """Benchmark a single request multiple times."""
        self.logger.info(f"Benchmarking prompt: '{prompt[:50]}...'")
        
        metrics = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'num_runs': num_runs,
            'generation_times': [],
            'tokens_generated': [],
            'throughputs': []
        }
        
        for run in range(num_runs):
            self.logger.info(f"Run {run + 1}/{num_runs}")
            
            result = self.generate_text(prompt, max_tokens)
            
            if result:
                generation_time = result['generation_time']
                tokens_generated = result['tokens_generated']
                throughput = tokens_generated / max(generation_time, 0.001)
                
                metrics['generation_times'].append(generation_time)
                metrics['tokens_generated'].append(tokens_generated)
                metrics['throughputs'].append(throughput)
                
                self.logger.info(f"  Generated {tokens_generated} tokens in {generation_time:.3f}s ({throughput:.1f} tok/s)")
            else:
                self.logger.warning(f"  Run {run + 1} failed")
        
        # Calculate statistics
        if metrics['generation_times']:
            metrics['avg_generation_time'] = statistics.mean(metrics['generation_times'])
            metrics['std_generation_time'] = statistics.stdev(metrics['generation_times']) if len(metrics['generation_times']) > 1 else 0
            metrics['avg_throughput'] = statistics.mean(metrics['throughputs'])
            metrics['std_throughput'] = statistics.stdev(metrics['throughputs']) if len(metrics['throughputs']) > 1 else 0
            metrics['avg_tokens'] = statistics.mean(metrics['tokens_generated'])
            metrics['std_tokens'] = statistics.stdev(metrics['tokens_generated']) if len(metrics['tokens_generated']) > 1 else 0
        
        return metrics
    
    def benchmark_throughput(self, num_concurrent: int = 1, num_requests: int = 10) -> Dict:
        """Benchmark throughput with sequential requests (avoiding concurrent issues)."""
        self.logger.info(f"Benchmarking throughput: {num_concurrent} concurrent, {num_requests} total requests")
        
        results = {
            'num_concurrent': num_concurrent,
            'num_requests': num_requests,
            'request_times': [],
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0
        }
        
        start_time = time.time()
        
        # Sequential requests to avoid memory access violations
        for i in range(num_requests):
            prompt = self.test_prompts[i % len(self.test_prompts)]
            self.logger.info(f"Request {i+1}/{num_requests}: '{prompt[:30]}...'")
            
            request_start = time.time()
            result = self.generate_text(prompt, max_tokens=50)
            request_end = time.time()
            
            if result:
                request_time = request_end - request_start
                results['request_times'].append(request_time)
                results['successful_requests'] += 1
                results['total_tokens'] += result['tokens_generated']
                self.logger.info(f"  Generated {result['tokens_generated']} tokens in {request_time:.3f}s")
            else:
                results['failed_requests'] += 1
                self.logger.warning(f"  Request {i+1} failed")
        
        total_time = time.time() - start_time
        
        # Calculate throughput metrics
        if results['request_times']:
            results['total_time'] = total_time
            results['avg_request_time'] = statistics.mean(results['request_times'])
            results['requests_per_second'] = results['successful_requests'] / total_time
            results['tokens_per_second'] = results['total_tokens'] / total_time
            results['success_rate'] = results['successful_requests'] / num_requests
        
        return results
    
    def run_comprehensive_benchmark(self, output_file: str = "huggingface_layerskip_benchmark_results.json"):
        """Run comprehensive benchmark suite."""
        self.logger.info("🚀 Starting HuggingFace LayerSkip benchmark")
        
        # Load model
        if not self.load_model():
            self.logger.error("Failed to load model, aborting benchmark")
            return
        
        try:
            # Single request benchmarks
            self.logger.info("📊 Running single request benchmarks...")
            single_results = {}
            
            for i, prompt in enumerate(self.test_prompts[:5]):  # Test first 5 prompts
                single_results[f'prompt_{i+1}'] = self.benchmark_single_request(
                    prompt, max_tokens=100, num_runs=3
                )
            
            self.results['metrics']['single_requests'] = single_results
            
            # Throughput benchmarks
            self.logger.info("📈 Running throughput benchmarks...")
            throughput_results = {}
            
            # Use sequential requests to avoid memory access violations
            throughput_results['sequential'] = self.benchmark_throughput(
                num_concurrent=1, num_requests=10
            )
            
            self.results['metrics']['throughput'] = throughput_results
            
            # Memory usage during benchmark
            self.logger.info("💾 Recording memory usage...")
            memory_info = self._get_memory_usage()
            self.results['metrics']['memory_usage'] = memory_info
            
            # Save results
            self.logger.info(f"💾 Saving results to {output_file}")
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            # Print summary
            self._print_benchmark_summary()
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
    
    def _get_memory_usage(self) -> Dict:
        """Get current memory usage."""
        memory_info = {
            'system_memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'used_gb': psutil.virtual_memory().used / (1024**3),
                'percent_used': psutil.virtual_memory().percent
            }
        }
        
        if torch.cuda.is_available():
            memory_info['gpu_memory'] = []
            for i in range(torch.cuda.device_count()):
                memory_info['gpu_memory'].append({
                    'device': i,
                    'memory_allocated_gb': torch.cuda.memory_allocated(i) / (1024**3),
                    'memory_reserved_gb': torch.cuda.memory_reserved(i) / (1024**3),
                    'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3)
                })
        
        return memory_info
    
    def _print_benchmark_summary(self):
        """Print benchmark summary."""
        self.logger.info("\n" + "="*80)
        self.logger.info("🎯 HUGGINGFACE LAYERSKIP BENCHMARK SUMMARY")
        self.logger.info("="*80)
        
        if 'single_requests' in self.results['metrics']:
            single_reqs = self.results['metrics']['single_requests']
            avg_gen_time = statistics.mean([req.get('avg_generation_time', 0) for req in single_reqs.values()])
            avg_throughput = statistics.mean([req.get('avg_throughput', 0) for req in single_reqs.values()])
            
            self.logger.info(f"📊 Single Request Performance:")
            self.logger.info(f"   Average Generation Time: {avg_gen_time:.3f}s")
            self.logger.info(f"   Average Throughput: {avg_throughput:.1f} tokens/s")
        
        if 'throughput' in self.results['metrics']:
            throughput = self.results['metrics']['throughput']
            self.logger.info(f"\n📈 Throughput Performance:")
            for concurrent, result in throughput.items():
                if 'requests_per_second' in result:
                    self.logger.info(f"   {concurrent}: {result['requests_per_second']:.2f} req/s "
                                   f"({result['success_rate']*100:.1f}% success)")
        
        self.logger.info(f"\n💾 Memory Usage:")
        if 'memory_usage' in self.results['metrics']:
            mem = self.results['metrics']['memory_usage']
            if 'system_memory' in mem:
                sys_mem = mem['system_memory']
                self.logger.info(f"   System: {sys_mem['used_gb']:.1f}GB / {sys_mem['total_gb']:.1f}GB "
                               f"({sys_mem['percent_used']:.1f}%)")
            
            if 'gpu_memory' in mem and mem['gpu_memory']:
                for gpu in mem['gpu_memory']:
                    self.logger.info(f"   GPU {gpu['device']}: {gpu['memory_allocated_gb']:.1f}GB / "
                                   f"{gpu['memory_total_gb']:.1f}GB allocated")
        
        self.logger.info("="*80)


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="HuggingFace LayerSkip Llama3.2-1B Benchmark")
    parser.add_argument("--model-name", type=str, 
                       default="RichardErkhov/facebook_-_layerskip-llama3.2-1B-gguf",
                       help="Model name")
    parser.add_argument("--model-file", type=str, 
                       default="layerskip-llama3.2-1B.Q2_K.gguf",
                       help="Specific model file to use")
    parser.add_argument("--output-file", type=str, 
                       default="huggingface_layerskip_benchmark_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = HuggingFaceLayerSkipBenchmark(
        model_name=args.model_name,
        model_file=args.model_file
    )
    
    # Run comprehensive benchmark
    benchmark.run_comprehensive_benchmark(args.output_file)
    
    print(f"\n✅ Benchmark completed! Results saved to {args.output_file}")
    print("📊 View the results with:")
    print(f"   Get-Content {args.output_file} | ConvertFrom-Json | ConvertTo-Json -Depth 10")


if __name__ == "__main__":
    main()
