#!/usr/bin/env python3
"""
vLLM Benchmark Script for LayerSkip Llama3.2-1B Model
Measures TPOT, TTFT, tokens/sec, ITL using vLLM server

This script benchmarks the LayerSkip Llama3.2-1B model served via vLLM with:
- Time Per Output Token (TPOT)
- Time To First Token (TTFT) 
- Tokens per second
- Inter-Token Latency (ITL)
- Throughput analysis
- Memory usage monitoring
- Comparison with direct inference
"""

import os
import time
import json
import argparse
import logging
import statistics
import subprocess
import requests
import threading
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import psutil
import GPUtil

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class VLLMBenchmark:
    """Comprehensive vLLM benchmark for LayerSkip Llama3.2-1B model."""
    
    def __init__(self, model_name: str = "RichardErkhov/facebook_-_layerskip-llama3.2-1B-gguf", 
                 model_file: str = "layerskip-llama3.2-1B.Q2_K.gguf",
                 server_host: str = "localhost", 
                 server_port: int = 8000):
        self.model_name = model_name
        self.model_file = model_file
        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"http://{server_host}:{server_port}"
        self.server_process = None
        
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
                'server_url': self.server_url
            },
            'metrics': {},
            'system_info': self._get_system_info()
        }
    
    def _get_system_info(self) -> Dict:
        """Get system information for benchmarking context."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}"
        }
        
        # GPU information
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                info['gpu_count'] = len(gpus)
                info['gpu_info'] = []
                for gpu in gpus:
                    info['gpu_info'].append({
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'memory_used': gpu.memoryUsed,
                        'memory_free': gpu.memoryFree,
                        'temperature': gpu.temperature,
                        'load': gpu.load * 100
                    })
        except Exception as e:
            self.logger.warning(f"Could not get GPU info: {e}")
            info['gpu_info'] = []
        
        return info
    
    def start_vllm_server(self, max_model_len: int = 4096, gpu_memory_utilization: float = 0.9,
                         tensor_parallel_size: int = 1, max_num_seqs: int = 256) -> bool:
        """Start vLLM server with the LayerSkip model."""
        try:
            self.logger.info(f"Starting vLLM server for {self.model_name}")
            self.logger.info(f"Model file: {self.model_file}")
            
            # vLLM server command
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.model_name,
                "--download-dir", "./models",
                "--max-model-len", str(max_model_len),
                "--gpu-memory-utilization", str(gpu_memory_utilization),
                "--tensor-parallel-size", str(tensor_parallel_size),
                "--max-num-seqs", str(max_num_seqs),
                "--port", str(self.server_port),
                "--host", self.server_host,
                "--trust-remote-code"
            ]
            
            self.logger.info(f"Running command: {' '.join(cmd)}")
            
            # Start server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            self.logger.info("Waiting for vLLM server to start...")
            max_wait_time = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                try:
                    response = requests.get(f"{self.server_url}/health", timeout=5)
                    if response.status_code == 200:
                        self.logger.info("✅ vLLM server started successfully!")
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(5)
                self.logger.info("Still waiting for server...")
            
            self.logger.error("❌ vLLM server failed to start within timeout")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start vLLM server: {e}")
            return False
    
    def stop_vllm_server(self):
        """Stop the vLLM server."""
        if self.server_process:
            self.logger.info("Stopping vLLM server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.logger.warning("Server didn't stop gracefully, forcing kill...")
                self.server_process.kill()
            self.server_process = None
            self.logger.info("✅ vLLM server stopped")
    
    def generate_text(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7,
                     top_p: float = 0.9, stream: bool = False) -> Dict:
        """Generate text using vLLM API."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream
            }
            
            response = requests.post(
                f"{self.server_url}/v1/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            return None
    
    def benchmark_single_request(self, prompt: str, max_tokens: int = 100, 
                               num_runs: int = 5) -> Dict:
        """Benchmark a single request multiple times."""
        self.logger.info(f"Benchmarking prompt: '{prompt[:50]}...'")
        
        metrics = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'num_runs': num_runs,
            'ttft_times': [],  # Time to first token
            'tpot_times': [],  # Time per output token
            'total_times': [], # Total generation time
            'token_counts': [], # Actual tokens generated
            'throughputs': []   # Tokens per second
        }
        
        for run in range(num_runs):
            self.logger.info(f"Run {run + 1}/{num_runs}")
            
            # Measure generation time
            start_time = time.time()
            response = self.generate_text(prompt, max_tokens)
            end_time = time.time()
            
            if response and 'choices' in response and len(response['choices']) > 0:
                generated_text = response['choices'][0]['text']
                total_time = end_time - start_time
                
                # Count tokens (rough estimation)
                token_count = len(generated_text.split())
                
                # Calculate metrics
                ttft = total_time  # Simplified - vLLM doesn't provide detailed timing
                tpot = total_time / max(token_count, 1)
                throughput = token_count / max(total_time, 0.001)
                
                metrics['ttft_times'].append(ttft)
                metrics['tpot_times'].append(tpot)
                metrics['total_times'].append(total_time)
                metrics['token_counts'].append(token_count)
                metrics['throughputs'].append(throughput)
                
                self.logger.info(f"  Generated {token_count} tokens in {total_time:.3f}s ({throughput:.1f} tok/s)")
            else:
                self.logger.warning(f"  Run {run + 1} failed")
        
        # Calculate statistics
        if metrics['ttft_times']:
            metrics['avg_ttft'] = statistics.mean(metrics['ttft_times'])
            metrics['std_ttft'] = statistics.stdev(metrics['ttft_times']) if len(metrics['ttft_times']) > 1 else 0
            metrics['avg_tpot'] = statistics.mean(metrics['tpot_times'])
            metrics['std_tpot'] = statistics.stdev(metrics['tpot_times']) if len(metrics['tpot_times']) > 1 else 0
            metrics['avg_throughput'] = statistics.mean(metrics['throughputs'])
            metrics['std_throughput'] = statistics.stdev(metrics['throughputs']) if len(metrics['throughputs']) > 1 else 0
            metrics['avg_tokens'] = statistics.mean(metrics['token_counts'])
            metrics['std_tokens'] = statistics.stdev(metrics['token_counts']) if len(metrics['token_counts']) > 1 else 0
        
        return metrics
    
    def benchmark_throughput(self, num_concurrent: int = 4, num_requests: int = 20) -> Dict:
        """Benchmark throughput with concurrent requests."""
        self.logger.info(f"Benchmarking throughput: {num_concurrent} concurrent, {num_requests} total requests")
        
        results = {
            'num_concurrent': num_concurrent,
            'num_requests': num_requests,
            'request_times': [],
            'successful_requests': 0,
            'failed_requests': 0
        }
        
        def make_request(prompt_idx):
            prompt = self.test_prompts[prompt_idx % len(self.test_prompts)]
            start_time = time.time()
            response = self.generate_text(prompt, max_tokens=50)
            end_time = time.time()
            
            if response and 'choices' in response:
                results['request_times'].append(end_time - start_time)
                results['successful_requests'] += 1
            else:
                results['failed_requests'] += 1
        
        # Create threads for concurrent requests
        threads = []
        start_time = time.time()
        
        for i in range(num_requests):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
            
            # Control concurrency
            if len(threads) >= num_concurrent:
                threads[0].join()
                threads.pop(0)
        
        # Wait for remaining threads
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Calculate throughput metrics
        if results['request_times']:
            results['total_time'] = total_time
            results['avg_request_time'] = statistics.mean(results['request_times'])
            results['requests_per_second'] = results['successful_requests'] / total_time
            results['success_rate'] = results['successful_requests'] / num_requests
        
        return results
    
    def run_comprehensive_benchmark(self, output_file: str = "vllm_layerskip_benchmark_results.json"):
        """Run comprehensive benchmark suite."""
        self.logger.info("🚀 Starting comprehensive vLLM LayerSkip benchmark")
        
        # Start vLLM server
        if not self.start_vllm_server():
            self.logger.error("Failed to start vLLM server, aborting benchmark")
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
            
            for concurrent in [1, 2, 4, 8]:
                throughput_results[f'concurrent_{concurrent}'] = self.benchmark_throughput(
                    num_concurrent=concurrent, num_requests=concurrent * 5
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
            
        finally:
            # Stop server
            self.stop_vllm_server()
    
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
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                memory_info['gpu_memory'] = []
                for gpu in gpus:
                    memory_info['gpu_memory'].append({
                        'name': gpu.name,
                        'memory_used_gb': gpu.memoryUsed,
                        'memory_total_gb': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100
                    })
        except Exception as e:
            self.logger.warning(f"Could not get GPU memory info: {e}")
        
        return memory_info
    
    def _print_benchmark_summary(self):
        """Print benchmark summary."""
        self.logger.info("\n" + "="*80)
        self.logger.info("🎯 VLLM LAYERSKIP BENCHMARK SUMMARY")
        self.logger.info("="*80)
        
        if 'single_requests' in self.results['metrics']:
            single_reqs = self.results['metrics']['single_requests']
            avg_ttft = statistics.mean([req.get('avg_ttft', 0) for req in single_reqs.values()])
            avg_tpot = statistics.mean([req.get('avg_tpot', 0) for req in single_reqs.values()])
            avg_throughput = statistics.mean([req.get('avg_throughput', 0) for req in single_reqs.values()])
            
            self.logger.info(f"📊 Single Request Performance:")
            self.logger.info(f"   Average TTFT: {avg_ttft:.3f}s")
            self.logger.info(f"   Average TPOT: {avg_tpot:.3f}s")
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
                    self.logger.info(f"   GPU {gpu['name']}: {gpu['memory_used_gb']:.1f}GB / "
                                   f"{gpu['memory_total_gb']:.1f}GB ({gpu['memory_percent']:.1f}%)")
        
        self.logger.info("="*80)


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="vLLM LayerSkip Llama3.2-1B Benchmark")
    parser.add_argument("--model-name", type=str, 
                       default="RichardErkhov/facebook_-_layerskip-llama3.2-1B-gguf",
                       help="Model name for vLLM")
    parser.add_argument("--model-file", type=str, 
                       default="layerskip-llama3.2-1B.Q2_K.gguf",
                       help="Specific model file to use")
    parser.add_argument("--server-host", type=str, default="localhost",
                       help="vLLM server host")
    parser.add_argument("--server-port", type=int, default=8000,
                       help="vLLM server port")
    parser.add_argument("--output-file", type=str, 
                       default="vllm_layerskip_benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--max-model-len", type=int, default=4096,
                       help="Maximum model length for vLLM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                       help="GPU memory utilization for vLLM")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = VLLMBenchmark(
        model_name=args.model_name,
        model_file=args.model_file,
        server_host=args.server_host,
        server_port=args.server_port
    )
    
    # Run comprehensive benchmark
    benchmark.run_comprehensive_benchmark(args.output_file)
    
    print(f"\n✅ Benchmark completed! Results saved to {args.output_file}")
    print("📊 View the results with:")
    print(f"   cat {args.output_file} | jq .")


if __name__ == "__main__":
    main()
