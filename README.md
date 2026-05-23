# TimesFM

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation
model developed by Google Research for time-series forecasting.

*   Paper:
    [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688),
    ICML 2024.
*   All checkpoints:
    [TimesFM Hugging Face Collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6).
*   [Google Research blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/).
*   TimesFM in Google 1P Products:
    *   [BigQuery ML](https://cloud.google.com/bigquery/docs/timesfm-model): Enterprise level SQL queries for scalability and reliability.
    *   [Google Sheets](https://workspaceupdates.googleblog.com/2026/02/forecast-data-in-connected-sheets-BigQueryML-TimesFM.html): For your daily spreadsheet. 
    *   [Vertex Model Garden](https://pantheon.corp.google.com/vertex-ai/publishers/google/model-garden/timesfm): Dockerized endpoint for agentic calling.

This open version is not an officially supported Google product.

> **Personal fork note:** I'm using this repo to experiment with TimesFM 2.5 for
> personal time-series projects (energy usage, finance). Feel free to open issues
> if you find bugs in my added examples.

**Latest Model Version:** TimesFM 2.5

**Archived Model Versions:**

-   1.0 and 2.0: relevant code archived in the sub directory `v1`. You can `pip
    install timesfm==1.3.0` to install an older version of this package to load
    them.

## Update - Apr. 9, 2026

Added fine-tuning example using HuggingFace Transformers + PEFT (LoRA) — see
[`timesfm-forecasting/examples/finetuning/`](timesfm-forecasting/examples/finetuning/).
Also added unit tests (`tests/`) and incorporated several community fixes.

Shoutout to [@kashif](https://github.com/kashif) and [@darkpowerxo](https://github.com/darkpowerxo). 

## Update - Mar. 19, 2026

Huge shoutout to [@borealBytes](https://github.com/borealBytes) for adding the support for [AGENTS](https://github.com/google-research/timesfm/blob/master/AGENTS.md)! TimesFM [SKILL.md](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting) is out.

## Update - Oct. 29, 2025

Added back the covariate support through XReg for TimesFM 2.5.


## Update - Sept. 15, 2025

TimesFM 2.5 is out!

Comparing to TimesFM 2.0, this new 2.5 model:

-   uses 200M parameters, down from 500M.
-   supports up to 16k context length, up from 2048.
-   supports continuous quantile forecast up to 1k horizon via an optional 30M
    quantile head.
-   gets rid of the `frequency` indicator.
-   has a couple of new forecasting flags.

Since the Sept. 2025 launch, the following improvements have been completed:

1.  ✅ Flax version of the model for faster inference.
2.  ✅ Covariate support via XReg (see Oct. 2025 update).
3.  ✅ Documentation, examples, and agent skill (see `timesfm-forecasting/`).
4.  ✅ Fine-tuning example with LoRA via HuggingFace Transformers + PEFT (see `timesfm-forecasting/examples/finetuning/`).
5.  ✅ Unit tests for core layers, configs, and utilities (see `tests/`).

### Install

1.  Clone the repository:
    ```shell
    git c
```
