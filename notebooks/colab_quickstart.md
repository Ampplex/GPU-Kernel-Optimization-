# Google Colab quick start

1. Open a Colab notebook.
2. Select **Runtime → Change runtime type → T4 GPU** (or another available supported NVIDIA GPU).
3. Clone the repository and install dependencies:

```python
!git clone https://github.com/Ampplex/GPU-Kernel-Optimization-.git
%cd GPU-Kernel-Optimization-
!pip install -r requirements.txt
```

Run correctness tests and benchmarks:

```python
!pytest -q
!python benchmarks/benchmark_gemm.py --m 1024 --n 1024 --k 1024 --dtype fp16
!python benchmarks/benchmark_attention.py --batch 1 --heads 4 --seq-len 512 --head-dim 64 --dtype fp16
```

Keep the GPU model, shapes, dtype, baseline latency, custom-kernel latency, and observed ratio in your experiment notes. Do not report a speedup until correctness has passed and measurements are reproducible.
