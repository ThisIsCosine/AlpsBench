# Data Usage Notice

This file covers the public benchmark data released in `benchmark_data/`.

## Scope

- public benchmark data:
  `benchmark_data/examples/`, `benchmark_data/dev/`, `benchmark_data/validation/`,
  `benchmark_data/test/`, and `benchmark_data/artifacts/`
- excluded private maintainer assets:
  `hidden/private_gold/` and any local unpublished tooling

## Provenance

As described in the paper, AlpsBench is derived from real-world human-LLM
dialogues curated from WildChat and paired with human-verified structured
memories. The paper describes a four-step pipeline of data collection, memory
extraction, human verification, and task construction.

## Data License

The paper states that WildChat is licensed under `ODC-BY` and that the released
reprocessed AlpsBench data is distributed under the same `ODC-BY` license for
academic and research use. Treat the benchmark data in this repository
accordingly.

## Redistribution Boundary

The current repository release treats `benchmark_data/` as the redistributable
public benchmark package.

- the public benchmark package may be hosted in this GitHub repository
- hidden evaluation references under `hidden/private_gold/` are not part of the
  public redistribution surface
- benchmark users should not attempt to reconstruct hidden gold from public
  files or run artifacts

## Privacy And Redaction

AlpsBench is built from real-dialogue-derived data. Use the public benchmark
package only for benchmark research and evaluation.

- do not use the data for deanonymization or identity linkage
- do not use the data to recover or infer hidden test references
- do not treat the public benchmark package as raw source logs
- for methodology and filtering details, follow the paper

## Citation

If you use AlpsBench, cite the paper referenced in [README.md](README.md).
