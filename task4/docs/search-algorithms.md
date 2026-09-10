# Search algorithms: exact KNN vs ANN with HNSW

## 1. What is being searched?

CLIP transforms every fashion image into a point in a 512-dimensional vector
space. Images that CLIP considers semantically or visually related tend to have
nearby directions in that space. The project L2-normalizes every vector:

```text
x_hat = x / ||x||₂
```

It then scores a stored vector `x_hat` against normalized query `q_hat` using:

```text
s(x_hat, q_hat) = x_hat · q_hat = cosine(x, q)
```

Cosine similarity normally lies from -1 to 1. Larger is more similar. CLIP
scores should be used for ranking, not interpreted as probabilities.

"KNN" describes the retrieval objective—find the K closest items. "ANN"
describes a family of algorithms that trade a controlled amount of exactness for
speed. In this codebase, the labels specifically mean:

- `knn`: exact exhaustive KNN with FAISS `IndexFlatIP`;
- `ann`: approximate KNN with FAISS `IndexHNSWFlat`.

## 2. Exact KNN (`IndexFlatIP`)

### Query procedure

For a query vector, exact search computes its inner product with every stored
vector, then selects the K largest values:

```text
scores = embeddings @ query
answer = indices of the K largest scores
```

FAISS implements this with optimized native vector operations and selection;
the Python code does not loop over images. Because every candidate is evaluated,
the returned neighbours are exact with respect to the stored vectors, metric,
and floating-point implementation.

### Cost

Let:

- `N` be the number of indexed images;
- `D` be embedding dimension (512 here);
- `K` be the requested result count.

The dominant work is `N` dot products of length `D`, approximately `O(ND)` per
query. Selection adds implementation-dependent Top-K work. Vector storage is
approximately `4ND` bytes for float32, excluding object/index overhead.

For this index:

```text
38,612 x 512 x 4 bytes = 79,077,376 bytes ≈ 75.4 MiB
```

That matches the approximate size of `embeddings.npy`. `IndexFlatIP` also holds
the vectors in memory, so loading the NumPy array and adding it to FAISS can
temporarily or persistently require another vector-sized allocation.

### Strengths

- Exact Top-K baseline: no algorithmic recall loss.
- No graph-training or tuning parameters.
- Simple and predictable at small-to-medium scale.
- Useful as ground truth when evaluating ANN quality.
- Can use a FAISS GPU when a compatible GPU-enabled FAISS installation exists.

### Limitations

- Query work grows linearly with catalog size.
- Memory bandwidth often becomes the bottleneck.
- Repeating an exhaustive scan is expensive for large catalogs or high query
  rates.

## 3. Approximate KNN with HNSW (`IndexHNSWFlat`)

### Graph intuition

Hierarchical Navigable Small World (HNSW) organizes vectors into a proximity
graph. Each vector is a node connected to selected nearby nodes. It maintains
multiple layers:

- the bottom layer contains all nodes and has dense local connectivity;
- progressively higher layers contain fewer randomly promoted nodes;
- upper layers provide long jumps across the vector space;
- lower layers refine the route near the query.

This resembles navigating a road system: highways first move toward the right
region, then local streets find close destinations. Search can avoid scoring
most of the catalog.

### Construction

When a vector is inserted, HNSW:

1. assigns it a maximum layer probabilistically;
2. starts at the current top entry point;
3. greedily descends toward closer nodes through upper layers;
4. searches a wider candidate set at each relevant layer;
5. selects and links nearby neighbours, pruning connections according to HNSW's
   graph heuristics.

This project calls:

```python
faiss.IndexHNSWFlat(D, m, faiss.METRIC_INNER_PRODUCT)
```

`Flat` means HNSW stores full float vectors: search is approximate because of
graph traversal, not because vectors are compressed or quantized. Once a
candidate is scored, its inner product uses the original stored float vector.

### Query procedure

Conceptually, HNSW search:

1. begins at an entry point in the highest layer;
2. repeatedly moves to a neighbour with a better query score;
3. descends a layer and continues from the best location found;
4. at the base layer, maintains a bounded pool of promising candidates;
5. stops after the configured exploration budget and returns its best K.

It can miss a true neighbour if the search path never reaches that node's local
region. This is the source of the accuracy/speed trade-off.

### Cost

HNSW often has sublinear and roughly logarithmic-like search behaviour on
well-behaved data, but `O(log N)` should not be treated as a universal guarantee.
Runtime depends on data geometry, graph quality, `M`, `efSearch`, dimension, and
hardware. Construction and memory usage are higher than a flat index because
the graph edges must also be created and stored.

### Parameters used here

#### `HNSW_M` / `m` (default 32)

`M` controls graph connectivity—the target number of bi-directional links per
node used by the HNSW construction. Increasing it generally:

- improves connectivity and recall, especially for difficult data;
- increases graph memory;
- increases index construction time;
- may increase query work because more neighbours can be inspected.

A lower value is leaner and faster to build but makes disconnected or poorly
navigable regions more likely. Changing `M` requires rebuilding the in-memory
HNSW index, but not regenerating CLIP embeddings.

#### `HNSW_EF_SEARCH` / `efSearch` (default 64)

`efSearch` controls the size of the dynamic candidate list explored at query
time. Increasing it usually improves recall but adds distance computations and
latency. It can be changed without regenerating embeddings or changing graph
connectivity.

In normal use, choose `efSearch >= K`. If K exceeds a small `efSearch`, the
algorithm has too little exploration budget to form a strong Top-K result set.

FAISS also has an HNSW construction parameter commonly called
`efConstruction`. The current code does not set it, so FAISS's version-specific
default is used.

## 4. Direct comparison

| Property | Exact KNN (`IndexFlatIP`) | ANN (`IndexHNSWFlat`) |
|---|---|---|
| Result guarantee | Exact Top-K for the stored vectors | May miss some exact Top-K items |
| Query strategy | Score every vector | Navigate a proximity graph |
| Query scaling | Linear in `N` for fixed `D` | Usually sublinear in practice; data/tuning dependent |
| Build cost | Low: add/copy vectors | Higher: build graph links |
| Memory | Full float vectors | Full float vectors plus graph |
| Main tuning | None beyond K/metric | `M`, `efSearch` (and potentially `efConstruction`) |
| Best role here | Accuracy baseline and modest catalogs | Lower-latency search as catalog/QPS grows |

For 38,612 vectors, exact search may already be entirely adequate. ANN becomes
more compelling when catalog size, concurrency, or latency requirements make an
exhaustive scan too costly. The choice should be based on measured end-to-end
latency and recall, not the algorithm label alone.

## 5. How to measure ANN quality correctly

Statements such as "HNSW has 99% recall" are meaningless without a dataset,
query set, K, and parameter configuration. Measure it against exact KNN:

1. choose a representative set of query images;
2. obtain exact result IDs `E_q` with `mode=knn`;
3. obtain approximate result IDs `A_q` with `mode=ann` at the same K;
4. compute per-query overlap:

```text
Recall@K(q) = |A_q ∩ E_q| / K
```

5. average over all queries and report latency percentiles (p50, p95, p99), not
   only a single query or mean;
6. sweep `M` and `efSearch`, including warm-up runs, until the quality/latency
   point meets the application requirement.

This recall measures agreement with exact CLIP-vector neighbours. It does not
prove fashion relevance. For that, use labels or human judgements and metrics
such as category Precision@K, mAP, or nDCG.

## 6. Timing caveat in this command-line application

`search.py` reports two different measurements:

- `Search time` surrounds only `store.search(...)` and is the appropriate local
  comparison of the already-built FAISS indexes;
- `Total time` includes loading embeddings, building the selected index, loading
  CLIP, and embedding the query.

Since HNSW is rebuilt on every command invocation, it can have a very small
reported search time but a larger total time than exact KNN for a single query.
For a fair online-system benchmark, build each index once, warm it up, then send
many identical query workloads through the long-lived process. Report startup
cost separately.

## 7. Practical selection guide

Use exact KNN when:

- the current catalog and query rate already meet the latency target;
- exact reproducibility of Top-K neighbours matters;
- you need a correctness baseline for retrieval experiments;
- operational simplicity is more valuable than marginal lookup speed.

Use HNSW ANN when:

- exact scan latency or throughput no longer meets requirements;
- the index will be reused for many queries, amortizing graph construction;
- extra graph memory is acceptable;
- measured Recall@K remains within the application's tolerance.

Tune in this order:

1. fix the desired K and representative evaluation queries;
2. select a reasonable `M` under the memory/build budget;
3. sweep `efSearch` to reach the recall target at the lowest acceptable latency;
4. repeat after major changes to catalog size, embedding model, or hardware.

## 8. Edge cases and invariants

- The model used for queries must match the model used to build the corpus.
- Every vector must be finite, float32-compatible, dimensionally consistent,
  and normalized for inner product to equal cosine similarity.
- Requesting K larger than the catalog can yield FAISS sentinel indices; the
  implementation filters negative indices, so it returns at most `N` results.
- Duplicate or near-duplicate images can occupy multiple Top-K positions because
  no deduplication step exists.
- A query image drawn from the indexed training folder will commonly retrieve
  itself at or near score 1.0. Exclude its ID when evaluating non-self retrieval.
- ANN output can vary with library version, insertion order, threading, or graph
  construction details; benchmark the exact deployed build.

