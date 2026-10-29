# Data

The benchmark uses the [MIT-BIH Arrhythmia Database, version 1.0.0](https://physionet.org/content/mitdb/1.0.0/),
downloaded from its public distribution. Follow the dataset's access and license
terms. The local download manifest stores the source URLs, file lengths and
locally computed SHA256 values. These hashes record the downloaded content;
they are not represented as independently supplied server checksums.

Preparation excludes paced records 102, 104, 107 and 217. It uses the channel
named MLII, including the reversed channel ordering in record 114. Expert beat
annotations are inputs, so the task is offline beat classification. An R-peak
detector is not evaluated.

Each waveform is `signal[peak-90:peak+162]`, giving 252 samples at 360 Hz.
Boundary and nonfinite windows are dropped. Each window is centred and divided
by its population standard deviation, with a floor of 1e-6. There is no filter
or test-fitted normalization. The processed array stores waveform, label,
record, peak and past-RR inputs.

The label mapping is fixed:

| Class | Annotation symbols |
|---|---|
| N | N, L, R, e, j |
| S | A, a, J, S |
| V | V, E |
| F | F |
| Q | /, f, Q |

The strict split holds DS1 records 106, 114, 119 and 209 out for validation and
removes DS1 record 201 because it shares subject identity with DS2 record 202.
The [database directory](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm)
documents the subject and lead details. The 17 remaining training records are
the natural federated clients. All 22 DS2 records remain in the test endpoint.

| Split | Beats | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|---:|
| Training | 40,142 | 36,723 | 421 | 2,582 | 408 | 8 |
| Validation | 8,897 | 7,490 | 395 | 1,008 | 4 | 0 |
| Test | 49,692 | 44,240 | 1,837 | 3,220 | 388 | 7 |

Records and beat indices are retained in each split manifest. The smaller
6,396- and 20,000-beat conditions sample class-stratified subsets from the same
training pool, retaining all 17 clients. They do not allocate equal numbers of
beats to patients. See the patient allocation CSVs for the exact distribution.

The reference archive contains derived predictions, labels, indices and model
weights, plus preprocessing/download metadata. Raw and processed ECG waveforms
are downloaded or generated locally, rather than redistributed in the archive.
