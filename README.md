# VWS-DMSC
  * Source code for NAACL 2019 paper [A Variational Approach to Weakly Supervised Document-Level Multi-Aspect Sentiment Classification].

## Requirements
    * Python3
    * tensorflow-gpu 1.6 -- 1.10
    * tqdm
    * munkres

## Usage
To predict the sentiment on tripadvisor/beer, run
```bash
python tripadvisor.py / beer.py --aspect asp
```
