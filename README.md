# SystolicMesh

A parameterised, tile-scalable systolic-array matrix-multiplication engine in SystemVerilog, computing **C = A × B** over IEEE-754 Float32.

**Dependencies:** Verilator · Python ≥ 3.10 · NumPy

---

## Architecture

![SystolicMesh Architecture](Mesh.png)

### Systolic Mesh

The top-level module is a grid of Systolic Arrays. The host writes A and B one matrix row per cycle (`HOST_WORDS = N`) into double-buffered staging memories; a broadcast copies each tile's operands into its array one tile row per cycle, all arrays fire together, and the results land in a double-buffered output SRAM read by the consumer.

The mesh is parameterised by `MATRIX_SIZE = N` and `TILE_SIZE = T`. By default (`COLLAPSE_K = 1`) each of the `(N/T)²` output tiles has one T×T array that runs the full depth N, so there are N² PEs and no reduction step. With `COLLAPSE_K = 0` the problem is split into `(N/T)³` T×T tiles, `N/T` depth slices per output tile, and the AccumulationUnit sums the slices in a log2(N/T) adder tree; that uses N³/T PEs for a few cycles less latency.

### Systolic Array

Each Systolic Array is a T×T grid of Processing Elements holding its A block (T×K) and B block (K×T) locally. On start, row r of A and column c of B enter r and c cycles late and every operand moves one PE per cycle, so the arrays are fully synchronous: no handshakes between PEs and no drain step. Results are read straight from the PEs.

### Processing Element

Each PE takes one product every cycle and computes

$$C_{ij} = \sum_k A_{ik} \cdot B_{kj}$$

The FP32 adder has a 5-cycle latency, so products rotate through six partial sums and are added pairwise at the end; the FP32 multiplier and adder come from the sibling `ArithmeticLibrary` and accept a new operation every cycle.

> [!NOTE]
> The array's numeric precision is determined entirely by the adder and multiplier modules sourced from `ArithmeticLibrary`. Swapping them out for alternative implementations (e.g. BFloat16, FP16, or integer) is sufficient to change the precision of the entire design — no other architectural changes are required. The `DATA_WIDTH` parameter must also be updated to match the bit-width of the new format (e.g. `DATA_WIDTH = 16` for FP16 or BFloat16).

### Convolution via im2col

Convolution is mapped onto the same matrix multiply by reformatting the input offline — input image patches are flattened into rows of A, and kernels into columns of B:

```
A[N×N]  row i  = patch_i.flatten()     (depth zero-padded to N)
B[N×N]  col j  = kernel_j.flatten()
C[N×N]  C[i,j] = dot(patch_i, kernel_j)
```

When N is a perfect square the tests use a √N×√N kernel so a patch fills the depth exactly; otherwise they use a 3×3 kernel over as many channels as fit and zero-pad the depth to N. Multi-output-channel and overlapping-stride variants are handled by extending B with multiple filter columns or splitting patches across batches respectively. The PE array is shared with matmul — there are no architectural differences between the two modes.

---

## Performance

All figures measured with Verilator, verified against a Float64 NumPy reference (relative tolerance ≤ 1%). Cycles are from start to result for one matrix; 15 random sets per configuration.

### Cycle counts

| Matrix | Tile | Cycles, collapse-k (default) | PEs | Cycles, depth slices (`COLLAPSE_K = 0`) | PEs |
|--------|------|------|------|------|------|
| 16×16  | 2×2  | 71   | 256  | 56   | 2 048  |
| 16×16  | 4×4  | 89   | 256  | 79   | 1 024  |
| 16×16  | 8×8  | 149  | 256  | 146  | 512    |
| 16×16  | 16×16| 365  | 256  | 365  | 256    |
| 32×32  | 2×2  | 87   | 1 024 | 61  | 16 384 |
| 32×32  | 4×4  | 105  | 1 024 | 84  | 8 192  |
| 32×32  | 8×8  | 165  | 1 024 | 151 | 4 096  |
| 32×32  | 16×16| 381  | 1 024 | 370 | 2 048  |
| 32×32  | 32×32| 1 197| 1 024 | 1 197 | 1 024 |
| 64×64  | 4×4  | 137  | 4 096 | 89  | 65 536 |
| 64×64  | 8×8  | 197  | 4 096 | 156 | 32 768 |
| 64×64  | 16×16| 413  | 4 096 | 375 | 16 384 |
| 64×64  | 32×32| 1 229| 4 096 | 1 202 | 8 192 |
| 64×64  | 64×64| 4 397| 4 096 | 4 397 | 4 096 |

Cycle counts are fully deterministic across random seeds — hardware completion time is data-independent.

### Scaling behaviour

**Tile size dominates latency.** All arrays fire together, so the cycle count is set by one array: its skew (about 2T) plus the depth it accumulates (N with collapse-k, T without) plus the final pairwise add. Smaller tiles finish sooner.

**Doubling N costs little.** At T=4, N=16 → 32 → 64 takes 89 → 105 → 137 cycles with collapse-k, because the extra work is spread over more arrays running in parallel.

### Tile size trade-off

| | Small tile (e.g. 2×2) | Large tile (e.g. 16×16) |
|---|---|---|
| **Latency** | Low ✓ | High ✗ |
| **Arrays** | Many ✗ | Few ✓ |
| **PEs, collapse-k** | N² either way | N² either way |

Very large meshes without collapse-k (16 384 PEs or more) take hours to compile at the default `-Os`; build them with `make OPT_FAST=-O0`.

---

## Simulation

### Single test

Generate a stimulus set manually, then compile and simulate:

```bash
python matmul_tests.py --list                          # see available tests
python matmul_tests.py --gen mm_random --matrix-size 16

python conv_tests.py --list                            # any N; perfect squares use a sqrt(N) kernel
python conv_tests.py --gen conv_random --matrix-size 16

make          # Verilator
make vcs      # VCS
```

### Regression

Stimulus generation, compilation, and simulation are all handled automatically. The testbench is compiled once per tile configuration and the binary is reused across all stimulus sets.

```bash
make regression MATRIX_SIZE=16
make regression MATRIX_SIZE=16 REGRESSION_GROUP=matmul   # matmul only
make regression MATRIX_SIZE=64 FAST=1                    # single tile, faster
```

Pass criterion: relative error ≤ 1% per element against a Float64 reference. Results are written to `testbenches/results/readiness/readiness_report.log`.
