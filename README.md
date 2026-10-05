# SystolicMesh

A parameterised, tile-scalable systolic-array matrix-multiplication engine in SystemVerilog. It computes **C = A × B (+ bias)** in fp32, bf16 or int8 (int32 accumulation) and streams sets back to back. It is the matrix engine of [SIENNA](https://github.com/SoHam-56/SIENNA).

**Dependencies:** Verilator 5 · a C++20 compiler (`--timing` needs coroutines) · Python ≥ 3.10 · NumPy

---

## Architecture

![SystolicMesh Architecture](Mesh.png)

### Systolic Mesh

The top-level module (`src/top/SystolicMesh.sv`) is a grid of systolic arrays, and sets flow through it back to back.

1. **Load.** The host writes A and B one matrix row per cycle (`HOST_WORDS = N`) into two staging banks. The last row may come with the start.
2. **Broadcast.** A full staging bank is copied into every array's free operand bank, one tile row per cycle.
3. **Compute.** The arrays feed and accumulate on their own.
4. **Reduce.** A reducer per output tile combines each finished set's partial sums. It writes the result to one of `RESULT_BANKS` (default 4) result banks, which the consumer reads and releases.

All four steps overlap: while one set computes, the next is loading and the previous one is being read out.

The mesh is parameterised by `MATRIX_SIZE = N` and `TILE_SIZE = T`:

| `COLLAPSE_K` | Arrays | PEs | Reducer work |
|---|---|---|---|
| 1 (default) | one T×T array per output tile, `(N/T)²`, each running the full depth N | N² | bias only |
| 0 | `N/T` depth slices per output tile, `(N/T)³` T×T arrays | N³/T | also sums the slices |

`COLLAPSE_K = 0` buys a few cycles of latency at a large cost in PEs.

Per set, with the start pulse:

| Control | Effect |
|---|---|
| `bias_valid_i` / `bias_i` | Adds `bias_i[c]` to every element of column c, inside the reducer |
| `partial_i` | Keeps the set's sums in the PEs, so the next set adds to them. A deep K is accumulated over several sets without leaving the mesh |
| `weight_cached_i` / `weight_tile_i` | Takes B from the weight cache (`WC_TILES` N×N tiles, two regions so one fills while the other is read), so the host sends only A |
| `pack_shift_i` | **Packed set.** B is block-diagonal with blocks of `b = N >> pack_shift_i` columns, so up to N/b independent small jobs share one set. Each PE skips the products outside its own block, so whatever sits off the blocks in B never reaches a result |

The consumer reads the oldest result one word at a time or through the wide read port (`WIDE_READ` words per cycle, one per consumer lane), then releases it.

### Systolic Array

Each Systolic Array is a T×T grid of Processing Elements with two operand banks. The next set's A block (T×K) and B block (K×T) are written while the current set feeds. Row r of A and column c of B enter r and c cycles late, and every operand moves one PE per cycle, so the arrays are fully synchronous. A queued set follows the previous one with no gap: row 0 takes k = 0 of the next set the cycle after k = K−1 of the current one.

### Processing Element

Each PE takes one product every cycle and computes

$$C_{ij} = \sum_k A_{ik} \cdot B_{kj}$$

The adder's latency is covered by rotating the products through U = min(K, 6) partial sums. Every K products the PE moves on to the next of `ACC_BANKS` (default 4) banks of partial sums. One set therefore accumulates while earlier ones finish and are read out, and the reducer adds a pixel's partial sums in its adder tree. The multipliers and adders come from the sibling [ArithmeticLibrary](https://github.com/SoHam-56/ArithmeticLibrary) and accept a new operation every cycle.

### Number formats

The format is a build parameter (`EXP_W`, `MAN_W`); every PE, the reducer and the result banks follow it:

| Format | Operands | Sums, bias and results | `EXP_W` / `MAN_W` |
|---|---|---|---|
| fp32 | IEEE binary32 | fp32 | 8 / 23 |
| bf16 | bfloat16 | bf16 | 8 / 7 |
| int8 | two's-complement int8 | int32 (`ACC_W = 32`) | 0 / 7 |

`mesh_model.py` is a bit-exact model of the mesh's arithmetic in each format. It covers the same operand order, the partial-sum slots and the reducer tree. The regression compares every output against it bit for bit.

### Convolution via im2col

Convolution is mapped onto the same matrix multiply by reformatting the input offline. Input image patches are flattened into rows of A, and kernels into columns of B:

```
A[N×N]  row i  = patch_i.flatten()     (depth zero-padded to N)
B[N×N]  col j  = kernel_j.flatten()
C[N×N]  C[i,j] = dot(patch_i, kernel_j)
```

When N is a perfect square, the tests use a √N×√N kernel so that a patch fills the depth exactly. Otherwise they use a 3×3 kernel over as many channels as fit and zero-pad the depth to N. Multi-output-channel and overlapping-stride variants extend B with more filter columns or split patches across batches. The PE array is the same for matmul and convolution.

---

## Performance

All figures are Verilator simulation cycle counts. Every output matched `mesh_model.py` bit for bit, across 11 matmul tests (23 sets each) and, at N ≥ 16, the convolution tests (19 sets each). N = 64 was run in fp32.

### Latency of one set (start to result written, cycles)

| N | T | fp32 | bf16 | int8 | PEs, collapse-k | fp32, depth slices (`COLLAPSE_K = 0`) | PEs, depth slices |
|---|---|---|---|---|---|---|---|
| 8  | 2  | 51   | 46   | 27   | 64    | | |
| 8  | 4  | 69   | 64   | 45   | 64    | | |
| 8  | 8  | 129  | 124  | 105  | 64    | | |
| 16 | 2  | 59   | 54   | 35   | 256   | 55  | 2 048 |
| 16 | 4  | 77   | 72   | 53   | 256   | 75  | 1 024 |
| 16 | 8  | 137  | 132  | 113  | 256   | 134 | 512   |
| 16 | 16 | 353  | 348  | 329  | 256   | 353 | 256   |
| 32 | 2  | 75   | 70   | 51   | 1 024 | | |
| 32 | 4  | 93   | 88   | 69   | 1 024 | | |
| 32 | 8  | 153  | 148  | 129  | 1 024 | | |
| 32 | 16 | 369  | 364  | 345  | 1 024 | | |
| 32 | 32 | 1 185 | 1 180 | 1 161 | 1 024 | | |
| 64 | 4  | 125  | | | 4 096 | | |
| 64 | 8  | 185  | | | 4 096 | | |
| 64 | 16 | 401  | | | 4 096 | | |
| 64 | 32 | 1 217 | | | 4 096 | | |
| 64 | 64 | 4 385 | | | 4 096 | | |

In bf16 and int8, the depth-slice builds at N = 16 take 50 / 70 / 129 / 348 and 24 / 43 / 106 / 329 cycles (T = 2 / 4 / 8 / 16). Cycle counts are fully deterministic across random seeds: hardware completion time does not depend on the data.

In fp32, start to result written is `3T + K + T² + LAT + 17` cycles, where:
- K is the depth each array multiplies: N with collapse-k, T with depth slices.
- U = min(K, 6) is the number of partial sums per PE.
- `LAT = 1 + 5·⌈log2(RP·U + 1)⌉` is the reducer tree over RP depth slices plus the bias.

The constant is made up of 1 to launch, T + 2 to broadcast and commit, 2 feed registers, 2(T − 1) skew, K − 1 products, 8 to multiply, 5 to add and 2 to flag the set as final. The T² reducer reads and LAT follow.

bf16's multiply and add are 5 cycles shorter in total, which makes every bf16 latency exactly 5 cycles below fp32. int8's integer multiply-add is shorter again.

### Throughput

The mesh alone accepts a new set every `max(T + 2, K, T²)` cycles:
- each array spends K cycles feeding a set;
- each reducer reads its T² pixels one per cycle.

In the full SIENNA pipeline at N = 16, T = 4 with a streaming host, a bias-and-ReLU set completes every 19 cycles. That is the host's one-row-per-cycle load (N + 3 cycles per set), at 431 FLOP per cycle, 84% of the mesh's 256 MAC slots.

`TB_SystolicMesh` also runs every configuration in two ways: 5 sets one at a time, then 5 sets streamed. Its consumer reads one result word per cycle, so there the stream is bound by N² reads per set:

| N | T | fp32: 5 sets serial | fp32: 5 sets streamed | streamed, per set |
|---|---|---|---|---|
| 16 | 4  | 1 934 | 1 387 | 277 |
| 16 | 16 | 3 314 | 1 663 | 333 |
| 32 | 4  | 5 934 | 5 259 | 1 052 |
| 32 | 32 | 11 394 | 6 351 | 1 270 |

### Scaling behaviour

**Tile size dominates latency.** All arrays run in lockstep, so a set's latency is made up of:
- one array's skew (about 2T);
- its depth (N with collapse-k, T without);
- the multiply and add latency;
- the reducer reading T² pixels.

Smaller tiles finish sooner and stream faster.

**Doubling N costs little latency.** At T = 4, going from N = 16 to 32 to 64 takes 77 → 93 → 125 cycles with collapse-k in fp32, because the extra work is spread over more arrays running in parallel.

### Tile size trade-off

| | Small tile (e.g. 2×2) | Large tile (e.g. 16×16) |
|---|---|---|
| **Latency** | Low ✓ | High ✗ |
| **Throughput** | K cycles per set ✓ | T² cycles per set ✗ |
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

The regression handles stimulus generation, compilation and simulation. It compiles the testbench once per tile configuration and reuses the binary across all stimulus sets.

```bash
make regression MATRIX_SIZE=16                                        # fp32, every tile size
make regression MATRIX_SIZE=16 REGRESSION_GROUP=matmul                # matmul only
make regression MATRIX_SIZE=32 REGRESSION_OPTS="--format int8 --tiles 4"
make regression MATRIX_SIZE=16 REGRESSION_OPTS="--format bf16 --collapse-k 0"
make regression MATRIX_SIZE=64 FAST=1                                 # single tile, faster
```

Pass criterion: every output equals `mesh_model.py` bit for bit in the build's format. Results are written to `testbenches/results/readiness/readiness_report_<fmt>_ck<k>.log`. The unit benches `TB_SystolicArray`, `TB_PE_int8` and `TB_PE_pack` check the array, the int8 PE and the packed PE's out-of-block skip on their own.
