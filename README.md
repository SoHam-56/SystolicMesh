# SystolicMesh

A parameterised, tile-scalable systolic-array matrix-multiplication engine in SystemVerilog, computing **C = A × B** over IEEE-754 Float32.

**Dependencies:** Verilator · Python ≥ 3.10 · NumPy

---

## Architecture

![SystolicMesh Architecture](Mesh.png)

### Systolic Mesh

The top-level module is a grid of Systolic Arrays, and sets flow through it back to back. The host writes A and B one matrix row per cycle (`HOST_WORDS = N`) into two staging banks; the last row may come with the start. Three parts run at once: a broadcaster copies a full staging bank into every array's free operand bank, one tile row per cycle; the arrays feed and accumulate on their own; and a reducer per output tile combines each finished set's partial sums and writes it to one of `RESULT_BANKS` (default 4) result banks, which the consumer reads and releases. A set may carry a bias row, which the reducer adds to every element of its column.

The mesh is parameterised by `MATRIX_SIZE = N` and `TILE_SIZE = T`. By default (`COLLAPSE_K = 1`) each of the `(N/T)²` output tiles has one T×T array that runs the full depth N, so there are N² PEs. With `COLLAPSE_K = 0` the problem is split into `(N/T)³` T×T arrays, `N/T` depth slices per output tile, and the reducer also sums the slices; that uses N³/T PEs for a few cycles less latency.

### Systolic Array

Each Systolic Array is a T×T grid of Processing Elements with two operand banks, so the next set's A block (T×K) and B block (K×T) are written while the current set feeds. Row r of A and column c of B enter r and c cycles late and every operand moves one PE per cycle, so the arrays are fully synchronous. A queued set follows the previous one with no gap: row 0 takes k = 0 of the next set the cycle after k = K−1 of the current one.

### Processing Element

Each PE takes one product every cycle and computes

$$C_{ij} = \sum_k A_{ik} \cdot B_{kj}$$

The FP32 adder has a 5-cycle latency, so products rotate through six partial sums. Every K products the PE moves on to the next of `ACC_BANKS` (default 4) banks of partial sums, so one set accumulates while earlier ones finish and are read out; the reducer adds a pixel's six partials in its adder tree. The FP32 multiplier and adder come from the sibling `ArithmeticLibrary` and accept a new operation every cycle.

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

All figures measured with Verilator, verified against a Float64 NumPy reference (relative tolerance ≤ 1%); 21 random sets per configuration. The one-set-at-a-time mesh these replace is at git tag `serial_mesh_v1`.

### Throughput

Sets stream at `max(N, T²)` cycles per set with collapse-k: the arrays need N cycles per set (one product per PE per cycle over the full depth), and each reducer reads its T² pixels one per cycle. At T = 4 that is one set every N cycles for N ≥ 16. Measured in the full SIENNA pipeline with a host that streams one row per cycle, a 256×256×64 matrix product runs at 32.0 cycles per set at N = 32 (every PE busy every cycle) and 17.4 at N = 16, where the consumer's per-set overhead shows.

### Latency of one set

| Matrix | Tile | Cycles, collapse-k (default) | PEs | Cycles, depth slices (`COLLAPSE_K = 0`) | PEs |
|--------|------|------|------|------|------|
| 16×16  | 2×2  | 59   | 256  | 55   | 2 048  |
| 16×16  | 4×4  | 77   | 256  | 75   | 1 024  |
| 16×16  | 8×8  | 137  | 256  | 134  | 512    |
| 16×16  | 16×16| 353  | 256  | 353  | 256    |
| 32×32  | 2×2  | 75   | 1 024 | 60  | 16 384 |
| 32×32  | 4×4  | 93   | 1 024 | 80  | 8 192  |
| 32×32  | 8×8  | 153  | 1 024 | 139 | 4 096  |
| 32×32  | 16×16| 369  | 1 024 | 358 | 2 048  |
| 32×32  | 32×32| 1 185| 1 024 | 1 185 | 1 024 |
| 64×64  | 4×4  | 125  | 4 096 | 85  | 65 536 |
| 64×64  | 8×8  | 185  | 4 096 | 144 | 32 768 |
| 64×64  | 16×16| 401  | 4 096 | 363 | 16 384 |
| 64×64  | 32×32| 1 217| 4 096 | 1 190 | 8 192 |
| 64×64  | 64×64| 4 385| 4 096 | 4 385 | 4 096 |

Depth-slice builds need a lot of memory: 65 536 PEs took 182 GB and almost 5 hours. The 16×16 T=2/T=4 and 32×32 T=4 depth-slice figures are 5 cycles above their first measurement: there the bias input makes the reducer's partial count one more than a power of two, which adds a tree level.

Every entry follows from the RTL: start to result written is `3T + K + T² + LAT + 17` cycles, where K is the depth each array multiplies (N with collapse-k, T with depth slices), U = min(K, 6) partial sums per PE, and `LAT = 1 + 5·⌈log2(RP·U + 1)⌉` is the reducer tree over RP depth slices plus the bias. The terms: 1 to launch, T + 2 to broadcast and commit, 2 feed registers, 2(T − 1) skew, K − 1 products, 8 multiply, 5 add, 2 to flag the set final, then T² reducer reads and LAT. Cycle counts are fully deterministic across random seeds — hardware completion time is data-independent.

### Scaling behaviour

**Tile size dominates latency.** All arrays run in lockstep, so a set's latency is one array's skew (about 2T), its depth (N with collapse-k, T without), the adder latency, and the reducer reading T² pixels. Smaller tiles finish sooner and stream faster.

**Doubling N costs little latency.** At T=4, N=16 → 32 → 64 takes 77 → 93 → 125 cycles with collapse-k, because the extra work is spread over more arrays running in parallel.

### Tile size trade-off

| | Small tile (e.g. 2×2) | Large tile (e.g. 16×16) |
|---|---|---|
| **Latency** | Low ✓ | High ✗ |
| **Throughput** | N cycles per set ✓ | T² cycles per set ✗ |
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
