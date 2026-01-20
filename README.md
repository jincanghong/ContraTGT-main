# ContraTGT: Contrastive Learning of Temporal Graph Transformer with Learnable Data Augmentation

## Introduction

## Requirements

* `python >= 3.7`, `PyTorch >= 1.4`, please refer to their official websites for installation details.
* Other dependencies:

```{bash}
pandas==0.24.2
tqdm==4.41.1
numpy==1.16.4
scikit_learn==0.22.1
matploblib==3.3.1
numba==0.51.2
```

## Dataset and preprocessing

#### Option 1: Use our preprocessed data

We provide preprocessed datasets: ia-slashdot-reply-dir, soc-wiki-elec, socialevolve_1month, and ubuntu. The dataset
address can be found in the paper.

You may check that each dataset corresponds to three files: one `.csv` ,Node features obtained by random generation
method.

```bash
# torch.nn.init.normal_(feature)
# torch.nn.init.uniform_(feature, a=-1.0, b=1.0)
# torch.nn.init.xavier_normal_(feature)
# torch.nn.init.xavier_uniform_(feature)
```

#### Option 2: process

Put your data under `data` folder. The required input data includes `ml_${DATA_NAME}.csv`, `${DATA_NAME}.content` . They
store the edge linkages and node features respectively.

```
u, i, ts, label, idx
```

All node index starts from `1`. The zero index is reserved for `null` during padding operations. So the maximum of node
index equals to the total number of nodes. Similarly, maxinum of edge index equals to the total number of temporal
edges. The padding embeddings or the null embeddings is a vector of zeros.

We also recommend discretizing the timestamps (`ts`) into integers for better indexing.

## Option 3: pre-train:

```bash
python pretrain.py -d slashdot --bs 800 --ctx_sample 30 --tmp_sample 21 --seed 60
```

## Training Commands

#### Examples:

* To train **ContraTGT** with Wikipedia dataset in inductive training, sampling 64 length-2 CAWs every node, and with
  alpha = 1e-5:

```bash
python main.py -d slashdot --bs 800 --ctx_sample 40 --tmp_sample 31 --seed 60
```

Note that the Socialevolve_1month dataset requires a smaller learning rate such as 3e-4

```bash
nohup python main.py --data slashdot --model_name CoLA_Former --gpu 1 --seed 60 --suffix slashdot_1_1 > slashdot_1_1.log &
```

```bash
nohup python main.py --data slashdot --model_name GraphMamba --bs 400 --gpu 1 --seed 60 --suffix slashdot_1_1 > slashdot_1_1.log &
```