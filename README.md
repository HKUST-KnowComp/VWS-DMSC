# VWS-DMSC
  * Source code for NAACL 2019 paper **A Variational Approach to Weakly Supervised Document-Level Multi-Aspect Sentiment Classification**

## Requirements
    * Python3
    * tensorflow-gpu 1.6 -- 1.10
    * tqdm
    * munkres

## Usage
To predict the sentiment on tripadvisor on aspect 0 (could be 0-6), run
```bash
python tripadvisor.py --aspect 0
```
To predict the sentiment on beer on aspect 0 (could be 0-3), run
```bash
python beer.py --aspect 0
```
To Run all aspects on both dataset
```bash
sh test.sh
```
