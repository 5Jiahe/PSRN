# This is the PyTorch Implementation of PSRN  
  
## Introduction  
This repository is Pytorch implementation of Progressive Semantic Reconstruction Network for Weakly Supervised Referring Expression Grounding.  
  
## Prerequisites  
  
* Python 3.6  
* Pytorch 1.9.0  
* CUDA 11.1  
  
## Installation  
  
1. Please refer to [MattNet](https://github.com/lichengunc/MAttNet) to install [mask-faster-rcnn](https://github.com/lichengunc/mask-faster-rcnn), [REFER](https://github.com/lichengunc/refer) and [refer-parser2](https://github.com/lichengunc/refer-parser2).  
Follow Step 1 & 2 in Training to prepare the data and features.  Place the proposal features extracted by mrcn in the directory cache/feats.
  
2. Download the data files needed for training, including [Glove](https://drive.google.com/drive/folders/1ksucJXMAUJ_NCZ5gEthLoMG9EtEJoqKz), semantic [similarity](https://drive.google.com/drive/folders/1h0Q02B-RgDCcWr2Hyh5nb8SZFW5mOMow) and subject and object [word](https://drive.google.com/drive/folders/1wFJYGCR7oP4y3Qrtl0yTCNfnTMsAfnUK). Place these files uniformly in cache. 
	You can also follow the steps below to generate these data yourself:
```bash  
python tools/gen_wds.py --dataset ${DATASET} --splitBy ${SPLITBY}  
python tools/cal_sim.py --dataset ${DATASET} --splitBy ${SPLITBY}  
```  
  
  
## Generating Key Triads  
  
Generate the Key Triads using Spacy:
```bash  
python tools/gen_triads.py --dataset ${DATASET} --splitBy ${SPLITBY}   
```
You can also download the generated key triads [file](https://drive.google.com/drive/folders/1nuiGKKnFyf0Qv1HYs7MmzmDPBZmcJjCV).
  
  
## Training  
  
Train PSRN :  
  
```bash  
python tools/train.py --dataset refcocog --splitBy google --stage 1  
```  
 You can change  ```--resume ```  to load the staged trained weights file for different modes of training.
 
## Evaluation  
  
Evaluate PSRN with trained weights:  
  
```bash  
python tools/eval.py --dataset refcocog --splitBy google --split val --stage 3
```  
 The trained weights file can be downloaded from [here](https://drive.google.com/drive/folders/1FT217iwNP268bVt-Emv5yhGUtOssSsP0),  place them in the directory output/.

## Acknowledge
This code is partially based on [DTWREG](https://github.com/insomnia94/DTWREG) and [MAttNet](https://github.com/lichengunc/MAttNet).
