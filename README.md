### Deep learning project seed
Use this seed to start new deep learning / ML projects.

- Built in setup.py
- Built in requirements
- Examples with MNIST
- Badges
- Bibtex

#### Goals
The goal of this seed is to structure ML paper-code the same so that work can easily be extended and replicated.

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT

---

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
<!-- ![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push) -->


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
 Next, navigate to any file and run it.
 ```bash
# module folder
cd src/efficient_face

# run module (example: mnist as your main contribution)
python train.py
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python

from flash import Trainer
from efficient_face.data.datasets import FaceRecognitionDataModule
from efficient_face.model import FaceRecognitionModel, SAMFaceRecognitionModel


# data
datamodule = FaceRecognitionDataModule.from_label_class_subfolders(
    train_folder="<path to data>",
    val_folder="<path to data>",
    batch_size=32,
)

# model
model = SAMFaceRecognitionModel()

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
