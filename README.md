# MIRA
This repository contains an adapted PyTorch implementation of an empathetic-response generation model, based on [MIME: MIMicking Emotions for Empathetic Response Generation](https://arxiv.org/pdf/2010.01454.pdf), with additional features for Russian language processing.

## Overview of MIRA
![Alt text](figs/MIRA.png?raw=true "Architecture of MIRA")

This repository presents a PyTorch implementation of an empathetic response generation model, adapted from the MIME approach with several architectural modifications and adjustments. This implementation extends MIME by incorporating small changes in model design and training setup to better handle Russian language data and improve empathy modeling. These adjustments enable a more nuanced generation of emotionally appropriate and contextually relevant responses, building upon the foundational ideas of polarity-based emotion clusters and emotional mimicry proposed in the original work.


## Setup
Download FastText vectors from [here](https://fasttext.cc/docs/en/crawl-vectors.html) and put it into `vectors/` folder

Next Install the required libraries:
1. Assume you are using conda
2. Install libraries by `pip install -r requirement.txt`

For reproducibility purposes, model output on the test dataset is provided as `./output.txt`.

## Run code
The dataset has been preprocessed using the same procedure as in the original work (https://github.com/HLTCHKUST/MoEL/tree/master/empathetic-dialogue) and is included in this repository. Additionally, the dataset was translated to Russian using the Yandex Translator API.

### Training
```sh
python main.py
```
> Note: This will also generate output file on test dataset as `save/test/output.txt`.

### Testing
```sh
python main.py --test --save_path [output_file_path]
```
> Note: During testing, the model will load weight under `save/saved_model`, and by default it will generate `save/test/output.txt` as output file on test dataset.

## Citation
If you use this work, please cite the original paper:
`MIME: MIMicking Emotions for Empathetic Response Generation. Navonil Majumder, Pengfei Hong, Shanshan Peng, Jiankun Lu, Deepanway Ghosal, Alexander Gelbukh, Rada Mihalcea, Soujanya Poria. EMNLP (2020).`

