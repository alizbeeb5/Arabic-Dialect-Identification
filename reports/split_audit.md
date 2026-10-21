# W1-07 split audit

**Split version:** `w1-07-pilot-v1-e61cd569ac7b`
**Split SHA256:** `e61cd569ac7b0bb3d7bd89a069ae82c7d02890deb99c3ef423c04a0f9a4d882f`
**Source release:** `full-8db5842080c2f643` (dataset revision `c8da454c9f474bb075d3fe608940ee29e8ccfa43`)
**Labels SHA256:** `5cf7d68237eab4a39790dda4faf59af6d31bff7c927564e55272e890206902c5`

Exact command:

```bash
python src/data/freeze_splits.py --manifest-dir src/data/manifests --dev-split dev --seed 490 --train-per-label 25 --dev-total 200
```

## 1. Split policy

- The upstream ADI-20 Micro split is preserved. `train` comes from upstream train; `dev` comes from the upstream validation split as mapped by `prepare.py`. No clip was moved between them.
- The development pool is **in-source**: same collection as training. It is not out-of-domain and must not be reported as such.
- No final-test data was read. Neither NADI 2026 test nor Casablanca test is an input to this script, so neither can influence tuning, pilot selection or model-generated labels.
- Grouping metadata (source IDs, hashes) was used only to build and audit splits. It is not a model input.

## 2. Grouping and separation

Clips were joined into one group whenever they shared any of: the filename-inferred `source_id`, the encoded-audio SHA256, or the decoded-PCM SHA256. Groups were formed transitively.

| Check | Result |
|---|---:|
| Clips read (train / dev) | 67015 / 10806 |
| Clips with unknown `source_id` | 70972 |
| Clips with known `speaker_id` | 0 |
| Groups | 71682 |
| Groups with more than one clip | 622 |
| Links via source_id / audio_sha256 / pcm_sha256 | 6128 / 11 / 11 |
| Groups crossing the train/dev boundary | 0 (0 clips) |
| Identical-audio groups with conflicting labels | 0 |

Groups crossing the boundary are listed in `reports/split_cross_groups.jsonl`. All of their clips were excluded from **both** pilot subsets. The upstream split itself still contains them; full-split training must decide separately whether to drop them.

**Limitations.** `source_id` is inferred from filenames and is unverified. No speaker IDs are available, so **speaker independence is not claimed**. Clips with unknown `source_id` are singleton groups unless an exact duplicate links them; recording overlap among those clips cannot be ruled out. Near-duplicate audio (re-encoded or trimmed copies) was not checked; only exact encoded and decoded matches were.

### Recording-ID coverage by dialect

A recording ID can only be inferred when the upstream filename has a YouTube ID. Clips with a known ID / all clips:

| Label | Train | Dev |
|---|---:|---:|
| MSA | 1705 / 3350 | 0 / 283 |
| BAH | 3350 / 3350 | 317 / 317 |
| TUN | 0 / 3350 | 1477 / 1649 |
| ALG | 0 / 2520 | 0 / 240 |
| EGY | 0 / 3222 | 0 / 650 |
| IRA | 0 / 3429 | 0 / 628 |
| JOR | 0 / 1913 | 0 / 367 |
| KSA | 0 / 3633 | 0 / 368 |
| KUW | 0 / 2847 | 0 / 430 |
| LEB | 0 / 3094 | 0 / 383 |
| LIB | 0 / 2550 | 0 / 627 |
| MAU | 0 / 2789 | 0 / 213 |
| MOR | 0 / 3031 | 0 / 380 |
| OMA | 0 / 4600 | 0 / 632 |
| PAL | 0 / 3129 | 0 / 430 |
| QAT | 0 / 4222 | 0 / 914 |
| SUD | 0 / 3846 | 0 / 197 |
| SYR | 0 / 3833 | 0 / 446 |
| UAE | 0 / 4414 | 0 / 1130 |
| YEM | 0 / 3893 | 0 / 522 |

A shared-recording check between train and dev is only possible where **both** splits have IDs: BAH.
- IDs on one side only: MSA, TUN. Nothing on the other side to compare against.
- No IDs in either split: ALG, EGY, IRA, JOR, KSA, KUW, LEB, LIB, MAU, MOR, OMA, PAL, QAT, SUD, SYR, UAE, YEM.

For every dialect outside the testable list, recording separation rests only on the upstream split and the exact-duplicate checks; it is not verified.

## 3. Pilot exclusions

| Reason | Clips |
|---|---:|
| (none) | 0 |

Per-clip records: `reports/split_exclusions.jsonl`.

## 4. Sampling procedure

1. Start from the eligible pools above.
2. For each label, sort its clips by `SHA256("w1-07-pilot-v1:<seed>:<clip_id>")`. This order depends only on the seed and clip IDs, not on Python's random module or the manifest row order.
3. Order that label's groups by the same seeded hash. Take clips round-robin: the first clip of every group, then the second clip of every group, and so on, until the quota is met. No group supplies a second clip while another eligible group has supplied none.
4. Train pilot: seed 490, 25 clips per label. Dev pilot: seed 491, 200 clips spread evenly over the 20 labels.

The exact IDs, seeds and hashes are recorded in `data/manifests/split_freeze.json`.

## 5. Class counts

| Label | Train pool | Train pilot | Dev pool | Dev pilot |
|---|---:|---:|---:|---:|
| MSA | 3350 | 25 | 283 | 10 |
| BAH | 3350 | 25 | 317 | 10 |
| TUN | 3350 | 25 | 1649 | 10 |
| ALG | 2520 | 25 | 240 | 10 |
| EGY | 3222 | 25 | 650 | 10 |
| IRA | 3429 | 25 | 628 | 10 |
| JOR | 1913 | 25 | 367 | 10 |
| KSA | 3633 | 25 | 368 | 10 |
| KUW | 2847 | 25 | 430 | 10 |
| LEB | 3094 | 25 | 383 | 10 |
| LIB | 2550 | 25 | 627 | 10 |
| MAU | 2789 | 25 | 213 | 10 |
| MOR | 3031 | 25 | 380 | 10 |
| OMA | 4600 | 25 | 632 | 10 |
| PAL | 3129 | 25 | 430 | 10 |
| QAT | 4222 | 25 | 914 | 10 |
| SUD | 3846 | 25 | 197 | 10 |
| SYR | 3833 | 25 | 446 | 10 |
| UAE | 4414 | 25 | 1130 | 10 |
| YEM | 3893 | 25 | 522 | 10 |
| **Total** | 67015 | 500 | 10806 | 200 |

### Shortfalls

None: every label met its quota.

### Spread across recording groups

Groups used / groups available, and the most clips taken from any one group. Unknown-source clips are singleton groups, so for those dialects every clip counts as its own group.

| Label | Train groups | Train max/group | Dev groups | Dev max/group |
|---|---:|---:|---:|---:|
| MSA | 25 / 1828 | 1 | 10 / 283 | 1 |
| BAH | 25 / 529 | 1 | 4 / 4 | 3 |
| TUN | 25 / 3341 | 1 | 10 / 175 | 1 |
| ALG | 25 / 2520 | 1 | 10 / 240 | 1 |
| EGY | 25 / 3222 | 1 | 10 / 650 | 1 |
| IRA | 25 / 3429 | 1 | 10 / 628 | 1 |
| JOR | 25 / 1913 | 1 | 10 / 367 | 1 |
| KSA | 25 / 3633 | 1 | 10 / 368 | 1 |
| KUW | 25 / 2847 | 1 | 10 / 430 | 1 |
| LEB | 25 / 3094 | 1 | 10 / 383 | 1 |
| LIB | 25 / 2550 | 1 | 10 / 627 | 1 |
| MAU | 25 / 2789 | 1 | 10 / 213 | 1 |
| MOR | 25 / 3031 | 1 | 10 / 380 | 1 |
| OMA | 25 / 4600 | 1 | 10 / 632 | 1 |
| PAL | 25 / 3129 | 1 | 10 / 430 | 1 |
| QAT | 25 / 4222 | 1 | 10 / 914 | 1 |
| SUD | 25 / 3846 | 1 | 10 / 197 | 1 |
| SYR | 25 / 3833 | 1 | 10 / 446 | 1 |
| UAE | 25 / 4414 | 1 | 10 / 1130 | 1 |
| YEM | 25 / 3893 | 1 | 10 / 522 | 1 |

## 6. Out-of-domain source

No separate-source development set is part of this freeze. Casablanca validation is the planned target-domain development set (`docs/evaluation_access.md`), but it has not been ingested or audited yet. **Week 2 task:** ingest Casablanca validation, recover episode grouping where possible, check recording and exact-duplicate overlap against ADI-20 training, and freeze it as the out-of-domain development set. Until then, every result on `dev_pilot` is in-source.

## 7. Acceptance checks

- [x] `train_pilot` is a subset of upstream train and `dev_pilot` a subset of upstream dev: every row was checked to be an unchanged row of its own upstream manifest.
- [x] Pilot subsets are disjoint by clip ID and by group (recording ID or exact-duplicate audio). Checked by the script before writing.
- [x] Class counts, exclusions, grouping limitations and the exact sampling procedure are recorded above.
- [x] The final test is excluded from tuning, pilot selection and model-generated training labels.
- [ ] A different teammate reviews this report and the generated manifests.

## 8. Input integrity

| File | SHA256 (verified against its sidecar) |
|---|---|
| train.jsonl | `56a75aa929e2a2c4592012954386cb3d91ff7f256a7004ebe7c4a776573ea4ba` |
| dev.jsonl | `525ecea4b39c504b28858ddbfaf0a954af703c91ae8edddb252a0a6f2c8d138b` |

Stored-audio verification used by the preparation run: `size`. Size checks cannot detect same-size corruption; rerun preparation with `--verification sha256` before final evaluation.
