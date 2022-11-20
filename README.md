<div align="center">

# EfficientFace

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->
<!-- [![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018) -->
<!-- [![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018) -->
<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)   -->
<!--
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
[![CI testing](https://github.com/karthikrangasai/efficient_face/actions/workflows/ci-testing.yml/badge.svg)](https://github.com/karthikrangasai/efficient_face/actions/workflows/ci-testing.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)


<!--
Conference
-->
</div>

## Description
Curation of various TripletLoss approaches described in Deep Learning Literature.

## How to run
First, install dependencies
```bash
# clone project
git clone https://github.com/karthikrangasai/efficient_face

# install project
cd efficient_face
pip install -e .
pip install -r requirements.txt
 ```
 <!-- Next, navigate to any file and run it.
 ```bash
# module folder
cd src/efficient_face

# run module (example: mnist as your main contribution)
python train.py
``` -->

## Usage
This project is setup as a package which means you can now easily import any file into any other file like so:
```python

from flash import Trainer
from efficient_face.data import ciFAIRDataModule
from efficient_face.model import TripletLossBasedTask, SoftmaxBasedTask


# data
datamodule = ciFAIRDataModule.load_ciFAIR10(batch_size=8)

# model
model = TripletLossBasedTask()

# train
trainer = Trainer()
trainer.fit(model, datamodule=datamodule)
```

<!-- ### Citation
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```    -->
