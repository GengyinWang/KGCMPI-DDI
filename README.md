# KGCMPI-DDI

Code for drug-drug interaction prediction using knowledge graph subgraph context and molecular association patterns.

## Setup Environment

```bash
conda create -n kgcmpi-ddi python=3.10
conda activate kgcmpi-ddi
```

Install compatible versions of [PyTorch](https://pytorch.org/get-started/locally/) and [DGL](https://www.dgl.ai/pages/start.html) for your system, then install the remaining dependencies:

```bash
pip install numpy scipy pandas scikit-learn tqdm rdkit dgllife
```

## Prepare Data

Place the processed datasets in the following directories:

```text
data/
├── DRKG+Deng/
└── DRKG+Ryu/
```

Each dataset directory should contain `nodes.tsv`, `edges.tsv`, `smiles.tsv`, and `ddi.tsv`, with consistent drug IDs across files. Use a header in `ddi.tsv` (`drug_left`, `drug_right`, `label`); the other files should have no header. Data is not included in the source archive.

## Running the Code

Run five-fold training and evaluation with the archive configuration:

```bash
python main.py --ddi_name Deng --condition S3 --apmp_parallel_fusion mean --fg_path_weight 1.0 --subgraph_gradient_checkpointing True
```

Use `--ddi_name Ryu` for the Ryu dataset, or `--condition S1` / `--condition S2` for other evaluation scenarios. Logs are saved in `log/`, and model checkpoints are saved in `data/DRKG+{dataset}/{scenario}/Fold{index}/`.
