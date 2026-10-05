# Hardware configuration — a2

## Host

| Field        | Value                |
|--------------|----------------------|
| Hostname     | a2                   |
| OS           | Ubuntu 24.04.3 LTS   |
| Kernel       | 6.8.0-100-generic    |
| Architecture | x86_64               |

## CPU

| Field              | Value                                   |
|--------------------|-----------------------------------------|
| Model              | Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz |
| Sockets            | 4                                       |
| Cores per socket   | 16                                      |
| Threads per core   | 1 (SMT disabled)                        |
| Total logical CPUs | 64                                      |
| Vendor             | GenuineIntel                            |

## Memory

| Field      | Value   |
|------------|---------|
| Total RAM  | 376 GiB |

## Storage

| Field      | Value          |
|------------|----------------|
| Root mount | /dev/sda2      |
| Capacity   | 2.1 TiB        |
| Used       | 242 GiB (12%)  |

## Notes

- CPU-only machine; no GPU present.  All d0bromir_gpu runs in benchmarks on this
  host exercise the CPU-fallback code path (fastp binary built with `make` or
  `make WITH_CUDA=1` but run without CUDA hardware).
- SMT is disabled (1 thread per core), so the logical CPU count (64) equals the
  physical core count.  Thread-budget fairness is directly verifiable: `-w N`
  maps to N physical cores.
