# Methods

## Learning workload

FedAvg runs for 30 rounds with one local epoch, batch size 128 and full client
participation unless a scenario specifies otherwise. The local AdamW optimizer
is reset for every client in every round, with learning rate 0.001, weight decay
0.0001 and gradient-norm clipping at 5. Aggregation uses participating-client
sample counts. Initialization is seeded. Local batch permutations use a NumPy
SeedSequence keyed by training seed, round, client and local epoch.
The `n_clients=10` setting applies to synthetic IID/Dirichlet partitions;
the patient partition creates one client per available training record, giving
17 clients in the primary split.

| Architecture | Definition | Parameters |
|---|---|---:|
| MLP | 252-128-64-5, ReLU hidden layers | 40,965 |
| MLP + RR | Waveform 252-128-64; timing 4-32-16; concatenation 80-64-5 | 46,837 |
| Compact CNN | Channels 1-16-32-64, kernels 7/5/5; temporal average; 64-64-5 head | 17,733 |
| Compact CNN + RR | Same encoder; append four RR features; 68-64-5 head | 17,989 |
| Spatial CNN + RR | Channels 1-16-32-32, kernels 7/5/3; flatten 32×31 to 64; timing 4-16; 80-64-5 head | 75,125 |

Convolutions have stride one. Compact CNN padding is 3/2/2; GroupNorm has
4/4/8 groups. Spatial CNN padding is 3/2/1; GroupNorm has 4/4/4 groups. Every
convolution block uses affine GroupNorm, ReLU and max-pooling with width/stride 2.
The compact encoder ends with adaptive averaging to one value per channel.
The spatial encoder retains 31 aligned positions. Hidden dense layers use ReLU.

RR inputs are the previous interval, second-previous interval, median of up to
ten preceding intervals and previous/median ratio. Values are clipped to
[0.1,5] and log-transformed. The first previous interval and empty median are
1 s; second-previous falls back to previous. The ratio denominator is floored
at 1e-3. No future RR interval is used, although the waveform extends after
the annotated peak.

## Local objectives and selection

`ce` is unweighted cross-entropy. `weighted` uses
`min(sqrt(n/max(n_k,1)),10)`. `balanced` uses
`min(n/max(n_k,1),100)`. `focal` uses the weighted rule with gamma=2.
Absent classes receive weight zero. Weights use pooled training-only counts,
are normalized to mean one over present classes, and are shared by clients.
Weighted cross-entropy and focal loss normalize by the sum of target weights
within a batch. They change the local objective; aggregation remains FedAvg.

The 16-recipe screen uses seeds 11, 22 and 33. Each run retains its highest
validation macro-F1 checkpoint. A strict score improvement is required, so a
round tie retains the earlier checkpoint. Recipe selection uses the mean of
those three peak validation values; no exact recipe-score tie occurred in the
saved screen. MLP/CE and the selected MLP+RR/balanced recipe are extended to
seeds 44 and 55.

Candidate families and weighting strengths were expanded during exploratory
development, and DS2 had already been inspected. Validation-only mechanical
selection and additional seeds do not create untouched test confirmation.
All tested candidates, including unsuccessful variants, are retained.

## Patient groups and training exposure

Grouped validation holds out these seven-patient groups in turn:

1. 208, 118, 220, 106, 112, 122, 230
2. 203, 205, 223, 207, 115, 119, 116
3. 101, 108, 109, 114, 124, 209, 215

Each model trains on the other 14 strict-DS1 patients. Round-30 predictions are
pooled over the 21 patients separately for each of three seeds. Training sets
overlap, so nine fold/seed cells are not nine independent replications. DS2 is
not evaluated in this stage. Class/patient support CSVs retain the rare-class
denominators.

One local epoch yields 1,800, 4,890 or 9,630 optimizer steps across 30 rounds
for 6,396, 20,000 or 40,142 training beats. The matched-exposure arms instead
both use `ceil(full_client_count/128)` steps per client, 128 replacement draws
per step and full-reference aggregation weights. Both receive 9,630 steps and
1,232,640 draws. The primary matched endpoint is fixed round 30.

The small subset starts with proportional floor class quotas, retaining at
least eight examples per class where available. Excess is removed from the
largest quota; deficits are filled by largest proportional shortfall. Sampling
within class is without replacement. The initial subset seed is 20261028;
20261029 and 20261030 add crossed sensitivity draws with learning seeds
11,22,33. Full-data models are reused across subset draws.

## Uncertainty

Macro-F1 averages the fixed five classes, assigning zero when a class has no
true or predicted positives. Accuracy, per-class precision/recall/F1 and support
are retained. A two-way paired bootstrap uses 5,000 repetitions:

1. Form one 5×5 confusion matrix per method, training seed and test patient.
2. Draw five seed indices and 22 patient indices independently with replacement.
   Share both index lists across methods, and the patient list across seeds.
3. Sum sampled patient matrices within each sampled seed, then compute fixed-
   five-class macro-F1. This pools beats before F1, rather than averaging patient F1.
4. Average over sampled seeds and take the paired method difference.
5. Use the 2.5th/97.5th percentiles with NumPy's linear interpolation.

Bootstrap seeds are 20261028 for the selected-recipe comparison and 20261029
for matched exposure. Crossed subset sensitivity additionally resamples three
subset indices and three training-seed indices, sharing draws across methods;
it averages over both dimensions after pooling patients, with seed 20261030.
The nine cells are three crossed subset draws and three seeds. Intervals
condition on fitted models and the explored recipes, without repeating model
selection or resampling training patients.

## Protected model transport

The library wrapper uses ML-KEM, HKDF-SHA256, AES-256-GCM and ML-DSA. The explicit
benchmark mapping is ML-KEM-512/ML-DSA-44, ML-KEM-768/ML-DSA-65 and
ML-KEM-1024/ML-DSA-87 for levels 1–3. Fresh key material is generated per message.
The existing 13-byte associated-data tag is retained exactly in the portable
implementation. These are fixed benchmark settings using standard algorithms.

Frozen MLP/CE and MLP+RR/balanced workloads use the strict 17-client training
set, three learning seeds and none/fixed-L3/adaptive policies. A separate RNG
with seed+20000 draws required levels with probabilities [0.244,0.712,0.044].
The same trace is used across directions, models and timing repeats. Adaptive
selection uses the minimum required level. This tests a bounded synthetic
trace, rather than every policy branch or a network deployment.

Every client receives a full FP32 global state and returns a full FP32 local
state each round. Received bytes are decoded into the model that is trained
or aggregated. All message payloads are compared byte-for-byte. Round hashes
are retained for every execution, with complete round-state arrays for one
timing repetition per workload, seed and policy.

Three isolated timing repetitions per learning seed yield 54 executions. Their
order is randomized before collection. All use one CPU thread, with shared
library loading and warm-up outside the timed rounds. Timing repeats are
averaged within seed, then the three seed means are summarized. No validation
or test prediction is used for this integration endpoint.

Serialized-component bytes include tensor data, public keys, encapsulation,
signature, nonce, authentication tag and associated data in both directions.
Per-message added lengths are 5,341 / 7,574 / 10,396 bytes for levels 1/2/3.
Framing, TLS, certificates, schema transfer and network latency are excluded.
The separate microbenchmark's `extra_bytes` omits the 13-byte associated-data
tag; the integration accounting explicitly adds it.

Host time includes sequential training, serialization, protection, decoding,
aggregation and audit instrumentation. Peak RSS includes the whole worker,
libraries, data loading and retained state vectors. These measurements are not
parallel-device latency or microcontroller memory/energy measurements. Optional
`resources` and `budget` modules use explicitly simulated bandwidth, power and
compute scaling; their energy outputs are model estimates.
