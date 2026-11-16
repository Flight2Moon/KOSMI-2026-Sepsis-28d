# EXP123 Config Profiles

- `experiment_cuda_max_gpu_throughput_7800x3d_48gb.json`: recommended for Ryzen 7 7800X3D, 64 GB RAM, RTX 5060 Ti 16 GB x2.
- `experiment_cuda_max_cpu_throughput_7800x3d_48gb.json`: compatibility alias for the same CUDA profile.
- `experiment_resource_cuda.json`: generic alias for the same validated CUDA profile.
- `experiment_resource_cpu.json`: required CPU reference profile used by tests and CPU-only runs.
- `experiment_cpu_7800x3d_48gb.json`: named CPU-only profile.

Use `run_exp123_cuda_dense.sh` for CUDA and `run_exp123_cpu_7800x3d.sh` for CPU-only execution.
