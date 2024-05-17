# NeuroTalk: Deep Generative Models for Speech Synthesis from Brain Signals

## Young-Eun Lee and Seong-Whan Lee

## Requirements
All algorithm are developed in Python 3.8.

To install requirements:

```setup
pip install -r requirements.txt
```

## Training
To train the model for spoken EEG in the paper, run this command:
```train
python train.py 
```
To train the model for Imagined EEG with pretrained model of spoken EEG in the paper, run this command:
```train
python train_fine_tuning.py 
```

## Inference
To evaluate the trained model for spoken EEG on an example data, run:
```inference
python inference.py 
```

## Pre-trained Models
You can download pretrained models here:
- [Pretrained model](https://drive.google.com/drive/folders/1_xh4OAI88yln73WNgjXg5EOqi6Ugltwh?usp=sharing) trained on participant 1

