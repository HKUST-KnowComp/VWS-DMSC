# VWS-DMSC
  * Source code for NAACL 2019 paper **A Variational Approach to Weakly Supervised Document-Level Multi-Aspect Sentiment Classification**

## Requirements
    * Python3
    * tensorflow-gpu 1.6 -- 1.10
    * tqdm
    * munkres

## Usage
To predict the polarity of all aspects on both dataset
```bash
sh test.sh
```
To predict the polarity of aspect 0 (could be 0-6) on tripadvisor
```bash
python tripadvisor.py --aspect 0
```
To predict the polarity of aspect 0 (could be 0-3) on beer
```bash
python beer.py --aspect 0
```

