# EMC-GCN

Code and datasets of our paper "[Enhanced Multi-Channel Graph Convolutional Network for Aspect Sentiment Triplet Extraction](https://aclanthology.org/2022.acl-long.212/)" accepted by ACL 2022.

## Requirements

- python==3.7.6

- torch==1.4.0
- transformers==3.4.0
- argparse==1.1

## Training

To train the EMC-GCN model, run:

```
cd EMC-GCN/code
sh run.sh
```
or
```
python main.py --mode train --dataset res14 --batch_size 16 --epochs 100 --model_dir savemodel/ --seed 1000 --pooling avg --prefix ../data/D2/
```

## Inference

To test the performance of EMC-GCN, you only need to modify the --mode parameter.
```
python main.py --mode test --dataset res14 --batch_size 16 --epochs 100 --model_dir savemodel/ --seed 1000 --pooling avg --prefix ../data/D2/
```

## Acknowledge

We appreciate all authors from this paper: "Grid Tagging Scheme for Aspect-oriented Fine-grained Opinion Extraction", because the code in this repository is based on their work [GTS](https://github.com/NJUNLP/GTS).

