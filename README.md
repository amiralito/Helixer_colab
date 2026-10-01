# Helixer Gene Prediction + BUSCO (Colab)

Batch *ab initio* gene prediction on Google Colab using
[Helixer](https://github.com/usadellab/Helixer) — a deep-learning gene caller (convolutional +
bidirectional-LSTM network predicting base-wise genic class and coding phase, decoded into gene
models by the HMM-based post-processor [HelixerPost](https://github.com/usadellab/HelixerPost)) —
with `gffread` CDS/protein extraction and built-in **BUSCO** completeness assessment. Point it at a
genome (or a whole folder of them) and it returns per-genome GFF3 + CDS + protein FASTA, a prediction
stats table, and BUSCO scores, all from a single notebook with form-field inputs.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amiralito/Helixer_Colab/blob/main/Helixer_gene_prediction.ipynb)

## Pipeline

```
genome FASTA  →   Helixer.py                          →   GFF3              →   gffread   →   BUSCO
(one or many)     fasta → H5 → CNN + biLSTM network        (gene, mRNA, exon,      CDS +         (proteins and/or
                  → HelixerPost (HMM decoding)             CDS, UTRs)              protein       genome mode)
                  lineage model from Zenodo                                        FASTA
```

## What it does

- Runs Helixer over **one or many genomes** (recursive folder scan, upload, or URL).
- Builds a **self-contained Helixer environment** beside Colab's (Python 3.10, TensorFlow 2.15.1,
  Keras 2, tensorflow-addons, CUDA 12.2 / cuDNN 8.9 pip wheels) and runs `Helixer.py` with that
  interpreter. Colab's own stack is never modified and **no session restart is needed**.
- Compiles **HelixerPost** (Rust) from source and puts `helixer_post_bin` in that environment.
- Downloads only the **lineage model** you pick from Helixer's live model list, caching it on Drive.
- Names outputs after each genome; gene IDs are `<SPECIES>_<seqid>_000001` (default `Helixer`).
- Extracts **CDS + protein FASTA** with `gffread`.
- **Gzips** all outputs and sorts them into `gff/`, `cds/`, `protein/`, `log/`, `busco/`, inside a
  **timestamped run folder** so consecutive runs never overwrite each other.
- **Resumable batches**: finished genomes are skipped (`SKIP_DONE`), a failed genome is logged and
  the batch carries on (`STOP_ON_ERROR` off), and `RESUME_RUN` reopens an interrupted run folder in a
  new session.
- Builds a per-genome **stats table** (genome size, genes, genes/Mb, mRNA, CDS, exons, exons/mRNA,
  UTRs, protein-length distribution, total CDS).
- Scores completeness with **BUSCO** in proteins and/or genome mode and **compiles a summary table**.
- All code cells are collapsed to **form fields** (`cellView: form`); mount Google Drive to persist
  inputs, models and outputs.

## Requirements

- Google Colab with a **GPU runtime** (`Runtime → Change runtime type → GPU`).
- **T4 (16 GB) is sufficient** with the default batch size for every lineage; **L4 / A100 / H100**
  are faster. A CPU runtime also works, but is only practical for a few Mb of sequence.
- **Blackwell** GPUs (compute capability 10.x / 12.x) postdate the pinned TensorFlow 2.15 / cuDNN 8.9;
  cell 3 runs a GPU smoke test and tells you if the card is unusable.
- **~5 GB of local disk** for the environment, rebuilt per session (~4–6 min).
- Genome FASTA only (`.fa/.fasta/.fna`, optionally `.gz`). No RNA-seq or protein evidence required.

## Quick start

1. Open the notebook in Colab and select a **GPU runtime**.
2. Run **cell 1** (runtime check), **cell 2** (install Helixer + HelixerPost + gffread), **cell 3** (verify).
3. **Cell 4** — optionally mount Drive, set `WORKDIR`, and stamp this run's folder (or resume one).
4. **Cell 5** — pick the lineage, set options, and choose the genome input mode.
5. **Cell 6** — download the model; **cell 7** — collect/stage the genome(s).
6. **Cell 8** — run Helixer + gffread; **cell 9** — view the stats table.
7. **Cells 10–13** — configure and run BUSCO, then compress the results.

## Notebook cells

| Cell | Step |
|------|------|
| 1 | Check runtime — GPU, memory, compute capability |
| 2 | Install Helixer (isolated Python 3.10 env, TF 2.15 + CUDA wheels), build HelixerPost, install gffread |
| 3 | Verify the install *inside that env* (versions, GPU, Conv1D + cuDNN-LSTM smoke test, HelixerPost, gffread) |
| 4 | Storage — mount Drive & stamp (or resume) a run folder |
| 5 | Prediction parameters & genome input |
| 6 | Download the lineage model from Zenodo (via Helixer's model list) |
| 7 | Collect genome(s) |
| 8 | Run Helixer + gffread → gzipped outputs into `gff/ cds/ protein/ log/` |
| 9 | Per-genome prediction stats table → `prediction_summary.tsv` |
| 10 | BUSCO options (mode toggles + lineage) |
| 11 | Install BUSCO (+ genome-mode tools if enabled) |
| 12 | Run BUSCO + compile `busco_summary.tsv` |
| 13 | Compress & save BUSCO run directories into `helixer_predictions/busco/` |

## Inputs & parameters (cell 5)

**Model** — `LINEAGE` selects the model; `MODEL_FILE = best` takes the top-priority model for that
lineage in Helixer's [model list](https://github.com/usadellab/Helixer/blob/main/resources/model_list.csv)
(cell 6 prints the full list). Type any other file name from that list to use it instead.

| Lineage | Best model (default) | Default `SUBSEQUENCE_LENGTH` |
|---|---|---|
| `land_plant` | `land_plant_v0.3_a_0080.h5` | 64152 (try up to 106920) |
| `vertebrate` | `vertebrate_v0.3_m_0080.h5` | 213840 |
| `invertebrate` | `invertebrate_v0.3_m_0100.h5` | 213840 |
| `fungi` | `fungi_v0.3_a_0100.h5` | 21384 |

**Species / IDs** (`SPECIES`, default `Helixer`) — written to the GFF3 `##species` header and used by
Helixer as the gene-ID prefix: `Helixer_<seqid>_000001`, transcripts `.1`. Leave empty to use each
genome's file name instead.

**Prediction**
- `SUBSEQUENCE_LENGTH` — how much sequence the network sees at once. `auto` = lineage default. It
  should comfortably exceed typical genic-locus length and stay below the assembly's N50 (ideally N90).
  Must be divisible by 9 (the model's timestep width).
- `OVERLAP` — sliding-window overlap (offset = length/2, core = 3/4 length) for better gene models at
  subsequence ends, at roughly twice the prediction time. On by default, as in `Helixer.py`.
- `BATCH_SIZE` — 32 (Helixer's default) fits a T4 for every lineage; lower it on OOM.
- `DETERMINISTIC` — deterministic cuDNN/cuBLAS kernels for bit-identical reruns on GPUs whose default
  kernels are not deterministic (e.g. L40S). Possibly slower.

**Post-processing (HelixerPost)**
- `PEAK_THRESHOLD` (0.8) — the precision/recall knob; 0.9, 0.95 and 0.975 give fewer spurious genes
  with very little BUSCO loss.
- `EDGE_THRESHOLD` (0.1), `WINDOW_SIZE` (100), `MIN_CODING_LENGTH` (60) — Helixer's defaults.

**Batch behaviour**
- `SKIP_DONE` — skip genomes that already have a GFF3 in this run folder.
- `STOP_ON_ERROR` — off by default: a failed genome is logged, listed at the end of cell 8, and the
  batch moves on.
- `WANT_CODING`, `WANT_PROTEIN` — extract CDS / protein FASTA with `gffread`.

**Genome source** (`GENOME_SOURCE`)
- `drive_dir` — run on **every** FASTA found recursively under `GENOME_DIR`.
- `drive_file` — a single genome already on Drive (`GENOME_PATH`).
- `url` — download a genome from `GENOME_URL`.
- `upload` — upload from your machine.

## Why a separate environment

Helixer 0.3.7 is pinned to **TensorFlow 2.15 / Keras 2** and imports `tensorflow-addons`, which was
discontinued and never published wheels for Python 3.12. Colab now ships Python 3.12 with TensorFlow
2.19 and Keras 3, so Helixer cannot be installed into the notebook kernel without downgrading
Colab's own stack.

Cell 2 therefore uses `uv` to fetch a standalone Python 3.10, creates `/content/helixer-env`,
installs Helixer's tested `requirements.3.10.txt` plus `tensorflow[and-cuda]==2.15.1` (CUDA 12.2 and
cuDNN 8.9 as pip wheels, independent of Colab's system CUDA), and cell 8 runs `Helixer.py` with that
interpreter. Colab's `PYTHONPATH` and inline matplotlib backend are stripped from the environment
Helixer sees, so no Python 3.12 package can leak in.

## Outputs

Each run writes to its own timestamped folder under `WORKDIR` (Drive when mounted, else the
ephemeral session). The models sit outside the run folders and are downloaded once:

```
WORKDIR/
├── helixer_models/<lineage>/<model>.h5        # cached model + model_list.csv, shared by every run
└── runs/
    └── run_20261001-142530[_tag]/             # stamped by cell 4
        ├── helixer_predictions/
        │   ├── gff/        <genome>_gff.gff3.gz          # Helixer's GFF3 (gene, mRNA, exon, CDS, UTRs)
        │   ├── cds/        <genome>_cds.fasta.gz
        │   ├── protein/    <genome>_protein.fasta.gz
        │   ├── log/        <genome>.log
        │   └── busco/      <genome>_busco_{prot,genome}_<lineage>.zip + busco_summary.tsv
        ├── prediction_summary.tsv   # per-genome: size, genes, genes/Mb, mRNA, exons, UTRs, protein lengths
        └── run_config.json          # model + md5, all parameters, genome list, package versions
```

The folder name is `run_<YYYYMMDD>-<HHMMSS>` in the timezone set in cell 4 (default
`Europe/London`), plus an optional `RUN_TAG` suffix. **Re-run cell 4 to start a new run folder**;
re-running cells 8–13 alone keeps writing into the current one. To continue an interrupted batch in a
new session, set `RESUME_RUN` in cell 4 to that folder's name — with `SKIP_DONE` on, finished genomes
are skipped and a `run_config_resumed_<time>.json` records the resumed session's settings.

## BUSCO (cells 10–13)

- **proteins mode** — scores the predicted annotation; needs only HMMER (fast).
- **genome mode** — scores the assembly via miniprot; independent of the annotation. Needs bbtools +
  miniprot and is CPU-bound, so it is much slower (consider an HPC for whole genomes).
- **Lineage** — pick a dataset matching your genomes' clade (plant, stramenopile/oomycete, fungal,
  animal, or generic `eukaryota`), or choose `other` and type any dataset (`busco --list-datasets`).
- **Several lineages** — every BUSCO output carries the lineage name, so they never collide. Change
  `BUSCO_LINEAGE` and re-run cells 10–13: `busco_summary.tsv` **accumulates** rows keyed by genome +
  mode + lineage (re-running a lineage replaces its rows rather than duplicating them).
- BUSCO always runs in a **local** directory (`/content/busco_runs`) because it uses symlinks that the
  Drive FUSE mount does not support; cell 13 zips each run into `helixer_predictions/busco/`.

## Notes & gotchas

- **Softmasked genomes need no preprocessing** — Helixer encodes lowercase and uppercase bases
  identically. Do not hardmask; the models were trained on unmasked sequence.
- **Helixer exits 0 even when post-processing fails**, so cell 8 treats a GFF3 without genes as a
  failure and points to the per-genome log.
- **GPU OOM** → lower `BATCH_SIZE` (and/or `SUBSEQUENCE_LENGTH`) in cell 5.
- **Cell 3 fails** → delete `/content/helixer-env` and re-run cell 2.
- **Fragmented assemblies** — keep `SUBSEQUENCE_LENGTH` below the N50/N90; long subsequences on short
  contigs waste compute on padding.
- **Genome basenames must be unique** across a `drive_dir` batch; cell 7 refuses duplicates.
- **Reproducibility** — predictions can differ slightly across GPU architectures regardless of
  `DETERMINISTIC`; `run_config.json` records the GPU-independent settings and the model md5.
- **Edit a hidden cell** → cell menu (⋮) → *Show code*.
- Results persist on Drive when mounted; otherwise the save cells offer downloads.

## Citation

If you use this notebook, please cite Helixer and the tools it depends on.

- **Helixer** — Holst F, Bolger AM, Kindel F, Günther C, Maß J, Triesch S, Kiel N, Saadat N,
  Ebenhöh O, Usadel B, Schwacke R, Weber APM, Bolger ME, Denton AK. *Helixer: ab initio prediction of
  primary eukaryotic gene models combining deep learning and a hidden Markov model.* Nat Methods. 2025.
  doi:10.1038/s41592-025-02939-1
- **Helixer (network)** — Stiehler F, Steinborn M, Scholz S, Dey D, Weber APM, Denton AK. *Helixer:
  cross-species gene annotation of large eukaryotic genomes using deep learning.* Bioinformatics.
  2020;36(22-23):5291–5298. doi:10.1093/bioinformatics/btaa1044
- **gffread** — Pertea G, Pertea M. *GFF Utilities: GffRead and GffCompare.* F1000Research. 2020;9:304.
  doi:10.12688/f1000research.23297.2
- **BUSCO** — Manni M, Berkeley MR, Seppey M, Simão FA, Zdobnov EM. *BUSCO Update: Novel and
  Streamlined Workflows along with Broader and Deeper Phylogenetic Coverage for Scoring of Eukaryotic,
  Prokaryotic, and Viral Genomes.* Mol Biol Evol. 2021;38(10):4647–4654. doi:10.1093/molbev/msab199
- **miniprot** (BUSCO genome mode) — Li H. *Protein-to-genome alignment with miniprot.*
  Bioinformatics. 2023;39(1):btad014. doi:10.1093/bioinformatics/btad014
- **OrthoDB** (BUSCO lineage datasets) — Kuznetsov D, Tegenfeldt F, Manni M, et al. *OrthoDB v11.*
  Nucleic Acids Res. 2023;51(D1):D445–D451. doi:10.1093/nar/gkac998
- **HMMER** — hmmer.org · **BBTools/BBMap** — Bushnell B., sourceforge.net/projects/bbmap

```bibtex
@article{holst2025helixer,
  author  = {Holst, Felix and Bolger, Anthony M. and Kindel, Felicitas and G{\"u}nther, Christopher and Ma{\ss}, Janina and Triesch, Sebastian and Kiel, Niklas and Saadat, Nima and Ebenh{\"o}h, Oliver and Usadel, Bj{\"o}rn and Schwacke, Rainer and Weber, Andreas P. M. and Bolger, Marie E. and Denton, Alisandra K.},
  title   = {{Helixer}: ab initio prediction of primary eukaryotic gene models combining deep learning and a hidden {Markov} model},
  journal = {Nature Methods},
  year    = {2025},
  doi     = {10.1038/s41592-025-02939-1}
}

@article{stiehler2020helixer,
  author  = {Stiehler, Felix and Steinborn, Marvin and Scholz, Stephan and Dey, Daniela and Weber, Andreas P. M. and Denton, Alisandra K.},
  title   = {{Helixer}: cross-species gene annotation of large eukaryotic genomes using deep learning},
  journal = {Bioinformatics},
  volume  = {36},
  number  = {22-23},
  pages   = {5291--5298},
  year    = {2020},
  doi     = {10.1093/bioinformatics/btaa1044}
}

@article{pertea2020gffread,
  author  = {Pertea, Geo and Pertea, Mihaela},
  title   = {{GFF} Utilities: {GffRead} and {GffCompare}},
  journal = {F1000Research},
  volume  = {9},
  pages   = {304},
  year    = {2020},
  doi     = {10.12688/f1000research.23297.2}
}

@article{manni2021busco,
  author  = {Manni, Mos{\`e} and Berkeley, Matthew R. and Seppey, Mathieu and Sim{\~a}o, Felipe A. and Zdobnov, Evgeny M.},
  title   = {{BUSCO} Update: Novel and Streamlined Workflows along with Broader and Deeper Phylogenetic Coverage},
  journal = {Molecular Biology and Evolution},
  volume  = {38},
  number  = {10},
  pages   = {4647--4654},
  year    = {2021},
  doi     = {10.1093/molbev/msab199}
}
```

## License & acknowledgements

This notebook is a convenience wrapper. Helixer and HelixerPost are released under the **GNU GPL v3**;
BUSCO, miniprot, gffread, HMMER and BBTools are the work of their respective authors and carry their
own licenses — please consult each upstream repository. Model weights are downloaded from Zenodo at
runtime via Helixer's model list.
