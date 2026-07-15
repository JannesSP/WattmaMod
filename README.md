# WattmaMod enables high-resolution and extensible RNA modification profiling for nanopore direct RNA sequencing

> Using WattMamba as the feature extractor, the pipeline combines contrastive learning with an anomaly-detection paradigm to enable self-training pretraining and supervised fine-tuning, supporting single-nucleotide detection of multiple RNA modifications, including m^6A, m^5C, m^1A, A-to-I, m^7G, hm^5C, m^1Ψ, f^5C, ac^4C, m^5U, and Ψ.
---

## 1. Overview

* **self-training pretraining (`train`)**: contrastive representation learning + anomaly pretraining on unlabeled DRS segments.
* **Single-stage fine-tuning (`fine`)**: train binary/multi-class classifiers on labeled data.
* **Two-stage fine-tuning (`two_fine`)**: domain adaptation / low-shot refinement on top of pretrained weights.
* **Evaluation (`test`)**: load the feature extractor and classifier to evaluate model performance on a held-out test set.
* **Prediction (`predict`)**: load trained weights to perform read-level prediction on input samples and output per-read modification probabilities, predicted labels, and related read-level results.

Core modules:

* `model/model`: backbone feature extractor (wavelet / cross-attention / SSM, etc.).
* `utils/anomaly_aug`: two data augmentation strategies for contrastive learning and anomaly pretraining.
* `main/train.py`: training and fine-tuning code.
* `main/main.py`: main entry script for argument parsing, configuration, and model execution.

For questions related to code usage, execution, or debugging, please contact: **[2023111001@mails.qust.edu.cn](mailto:2023111001@mails.qust.edu.cn)**

---

## 2. Dependencies

* Python ≥ 3.9
* PyTorch ≥ 2.1
* Mamba
* Uncalled4 = 4.1.0
* Others: `numpy`, `scipy`, `pandas`, `tqdm`
* Linux environment required

Install dependencies from `requirements.txt` first:

### using pip

```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt --index-url https://download.pytorch.org/whl/cu118 # this required cuda!
pip install causal-conv1d==1.5.0.post8 --no-build-isolation
pip install mamba-ssm==2.2.4 --no-build-isolation
```

Additional note: the Mamba environment needs to be configured manually.

A CUDA-capable NVIDIA GPU is required to run the software. CPU-only execution is not supported due to the Mamba-related configuration.

Typical installation time is approximately 10–30 minutes on a normal desktop computer, depending on the system environment and network conditions.

### using conda (TODO: fix this documentation)

install PyTorch

```bash
conda install cudatoolkit==11.8 -c nvidia   # CUDA 11.8 is required
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
```

# install extra packages for Mamba (not conda mamba)

This requires cuda!

```bash
pip install causal-conv1d
pip install mamba-ssm
```
Offline installation is recommended for faster setup:

* `causal-conv1d`: [https://github.com/Dao-AILab/causal-conv1d/releases/](https://github.com/Dao-AILab/causal-conv1d/releases/)
* `mamba_ssm`: [https://github.com/state-spaces/mamba/releases/](https://github.com/state-spaces/mamba/releases/)

---

## 3. Suggested Directory Layout
```text
WattMamba/
├── model/                         # Core model definitions (WattMamba, AnomalyDetectionModel)
├── save_model/                   # Saved model artifacts
│   ├── RNA002_model/                
│   └── RNA004_model/                
├── main/                    # Training & evaluation scripts
│   ├── anomaly_detection.py
│   ├── fine_tune_loss.py
│   ├── main.py
│   ├── predict.py
│   └── train.py
│
└── utils/                         # Tools and data processing modules
    │── anomaly_aug.py
    │── contrastive_loss.py
    │── dataloader.py
    │── tombo_fast5_to_signal.py
    │── tombo_signal_to_feature.py
    │── raw_signal_feature.py
    │── uncalled4_pod5_to_signal.py
    │── uncalled4_signal_to_feature.py
    └── transcriptome_genome_location.py
```

## 4. Data Preparation

This repository expects feature tables in **TSV** format as model input. Raw nanopore files should be preprocessed into `*.signal.tsv` and then converted into `*.feature.tsv` before pretraining, fine-tuning, evaluation, or prediction.

### 4.1 RNA004 samples

For RNA004 samples, the raw input is typically in **POD5** format. The recommended preprocessing workflow is:

* **Basecalling and reference-guided alignment with Dorado**

```bash
dorado basecaller rna004_130bps_sup@v5.1.0 \
  m6A/ \
  -x cuda:0 --emit-moves --emit-sam \
  -o m6A/ \
  --reference curlcake_reference.fa \
  --mm2-opts "-x map-ont" > m6a.bam
```

* **Signal-to-reference alignment with Uncalled4**

```bash
uncalled4 align --ref curlcake_reference.fa \
  -p 4 \
  --reads m6A/ \
  --bam-in m6a.bam \
  --rna \
  -o align.m6A.bam
```

* **Convert POD5 + aligned BAM into signal-level TSV**

```bash
python utils/uncalled4_pod5_to_signal.py \
  --pod5 m6A/ \
  --bam align.m6A.bam \
  --out m6A.signal.tsv \
  --ref curlcake_reference.fa
```

* **Convert signal-level TSV into feature-level TSV**

```bash
python utils/uncalled4_signal_to_feature.py \
  --signal m6A.signal.tsv \
  --out m6A.feature.tsv \
  --motif NNANN
```

If the raw files are in **FAST5** format instead of POD5, convert them to POD5 first:

```bash
pod5 convert fast5 \
  IVET_m5C \
  -o IVET_m5c/m5c.pod5 \
  -r \
  -t 16
```

### 4.2 RNA002 samples

For RNA002 samples, the raw input is usually in **FAST5** format. The recommended preprocessing workflow is:

* **Basecalling with Guppy**

```bash
guppy_basecaller \
  -i IVET_m6A \
  -s IVET_m6A_guppy \
  --num_callers 20 \
  --recursive \
  --fast5_out \
  --config rna_r9.4.1_70bps_hac.cfg \
  --device cuda:0
```

* **Convert multi-FAST5 to single-FAST5**

```bash
multi_to_single_fast5 \
  -i IVET_m6A_guppy \
  -s IVET_m6A_guppy_single \
  -t 10 \
  --recursive
```

* **Resquiggle with Tombo**

```bash
tombo resquiggle \
  --overwrite \
  --basecall-group Basecall_1D_000 \
  IVET_m6A_guppy_single \
  IVET_reference.fa \
  --processes 20 \
  --fit-global-scale \
  --include-event-stdev
```

* **Merge FASTQ reads**

```bash
cat IVET_m6A_guppy/pass/*.fastq > IVET_m6A.fastq
```

* **Map reads to reference**

```bash
minimap2 -ax map-ont IVET_reference.fa IVET_m6A.fastq > IVET_m6A.sam
```

* **Extract signal-level TSV**

```bash
python utils/tombo_fast5_to_signal.py \
  -p 20 \
  --fast5 IVET_m6A_guppy_single \
  --reference IVET_reference.fa \
  --sam IVET_m6A.sam \
  --output IVET_m6A.signal.tsv \
  --clip 10
```

* **Convert signal-level TSV into feature-level TSV**

```bash
python utils/tombo_signal_to_feature.py \
  --signal_file IVET_m6A.signal.tsv \
  --clip 10 \
  --output IVET_m6A.feature.tsv \
  --motif NNANN
```

### 4.3 Model input

After preprocessing, the repository uses **TSV** files as direct model input.

* `*.signal.tsv`: signal-level intermediate file extracted from POD5/FAST5 and alignment results.
* `*.feature.tsv`: feature-level file used for downstream pretraining, fine-tuning, evaluation, and prediction.

### 4.4 Notes

* RNA004 samples are processed with **Dorado + Uncalled4**.
* RNA002 samples are processed with **Guppy + Tombo**.
* If the raw RNA004 data are stored in FAST5 format, they should be converted to POD5 before downstream processing.
* The motif argument in feature extraction can be adjusted according to the target modification type, for example `NNANN` for m6A-centered candidate construction.


## 5. Usage

After obtaining the modified and corresponding unmodified `*.tsv` files, the model can be used for self-supervised pretraining, supervised fine-tuning, two-stage supervised fine-tuning, evaluation, and prediction.

The entry script exposes five `--run` modes: `train | fine | two_fine | test | predict`.

### 5.1 Self-supervised pretraining

Self-supervised pretraining is performed on unlabeled or weakly labeled DRS feature tables to initialize the backbone model.

```bash
python main.py \
  --run train \
  --pretrain_data1 /path/to/pre.unmod.a.tsv \
  --pretrain_data2 /path/to/pre.m6a.tsv \
  --pretrain_model_path /path/to/pre_trained.pth
```

Output: pretrained backbone weight file in `save_result/` (exact filename depends on `main/train.py`).

### 5.2 Single-stage supervised fine-tuning

After pretraining, supervised fine-tuning is performed using paired modified and unmodified `*.tsv` files.

```bash
python main.py \
  --run fine \
  --pretrain_model_path /path/to/pre_trained.pth \
  --train_data_mod /path/to/fine.ivet.m5c.tsv \
  --train_data_unmod /path/to/fine.ivet.unmod.c.tsv \
  --modified m5c 
```

Output: two weight files will be generated, including `backdone_m5c.pth` and `classifier_m5c.pth`.

### 5.3 Two-stage supervised fine-tuning

Two-stage fine-tuning further refines the pretrained and fine-tuned model on a new dataset or low-shot target dataset. In this stage, the backbone weight file and classifier weight file are both loaded for continued training.

```bash
python main.py \
  --run two_fine \
  --pretrain_model_path /path/to/backdone_m5c.pth \
  --finetune_model_path /path/to/classifier_m5c.pth \
  --train_data_mod /path/to/fine.hm5c.tsv \
  --train_data_unmod /path/to/fine.unmod.c.tsv \
  --modified hm5c 
```

Output: updated backbone and classifier weight files for the target modification task.

### 5.4 Evaluation

The evaluation mode loads the trained backbone and classifier to test model performance on a held-out dataset.

```bash
python main.py \
  --run test \
  --pretrain_model_path /path/to/backdone_m1a.pth \
  --finetune_model_path /path/to/classifier_m1a.pth \
  --test_data_mod /path/to/test.ivet.m1a.tsv \
  --test_data_unmod /path/to/test.ivet.unmod.a.tsv \
```

Metrics and logs are printed during evaluation.

### 5.5 Prediction

The prediction mode loads the trained model weights and performs read-level prediction on input samples.

```bash
python predict.py \
  --pre /path/to/backdone_m6a.pth \
  --fine /path/to/classifier_m6a.pth \
  --input /path/to/test.m6a.tsv \
  --o /path/to/predict.m6a.tsv 
```

Output: read-level prediction results, including per-read modification probabilities, predicted labels, and related prediction information.

---

## 6. Minimal Reproducible Workflow

### 6.1 Data preprocessing

For end-to-end data preprocessing, please refer to the following notebooks:

* `processing_RNA004.ipynb`: preprocessing workflow for RNA004 samples
* `processing_RNA002.ipynb`: preprocessing workflow for RNA002 samples
* `samples/`: an example directory containing RNA004 sample data and intermediate files

For RNA002 samples, signal extraction from FAST5 files may be relatively slow. In this case, you can modify the path of `sequencing_summary.txt` in `utils/extract_signal_from_fast5_to_temp.py` to accelerate preprocessing. The corresponding extraction strategy is adapted from the TandemMod GitHub workflow.

### 6.2 Minimal workflow example

```bash
python main.py \
  --run test \
  --pretrain_model_path RNA002_model/backbone_hm5C.pth \
  --finetune_model_path RNA002_model/classifier_hm5c.pth \
  --test_data_mod test.hm5c.tsv \
  --test_data_unmod test.hm5c.tsv \
  --batch_size 512 \
  --gpu 0
python predict.py \
  --input feature.tsv \
  --pre RNA004_model/backbone_hm5C.pth \
  --fine RNA004_model/classifier_hm5c.pth \
  --o predict.result.tsv \

```
