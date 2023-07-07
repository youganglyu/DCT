# Feature-Level Debiased Natural Language Understanding (AAAI 2023)
This repository is the implementation of our AAAI 2023 Paper [Feature-Level Debiased Natural Language Understanding](https://arxiv.org/abs/2212.05421). Please contact Yougang Lyu (youganglyu@gmail.com) if you have any question.

## Datasets and Checkpoints

Download the processed dataset and checkpoints from the [Google Drive](https://drive.google.com/drive/folders/1qy_h-mw03_jb8GHArbmSG9C4BZ95eIC_?usp=sharing).
The downloaded datasets should be moved into `/PATH_TO_DATA_DIR`.

The downloaded ckpt files should be moved into `/PATH_TO_OUTPUT_DIR`.

## Quick Start

To train the DCT model, run:

```
sh scripts/mnli_dct_train.sh #bert_path
sh scripts/fever_dct_train.sh #bert_path
sh scripts/snli_dct_train.sh #bert_path
```

You can also test the model has been saved by us.

```
sh scripts/mnli_dct_eval.sh #checkpoint_path
sh scripts/fever_dct_eval.sh #checkpoint_path
sh scripts/snli_dct_eval.sh #checkpoint_path
```

## Bias Extractability

The code for evaluating the extractability of biased features in the model representation is https://github.com/technion-cs-nlp/bias-probing.

## Citation

If you find our work useful, please cite our paper as follows:

```
@article{lyu-etal-2023-dct, 
title={Feature-Level Debiased Natural Language Understanding}, 
author={Yougang Lyu and Piji Li and Yechang Yang and Maarten de Rijke and Pengjie Ren and Yukun Zhao and Dawei Yin and Zhaochun Ren},
year={2023}, 
volume={37}, 
pages={13353-13361} }
```
