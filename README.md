# OSKut (Out-of-domain StacKed cut for Word Segmentation) 

[![](https://img.shields.io/badge/license-MIT-green)](https://github.com/mrpeerat/OSKut/blob/main/LICENSE)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrpeerat/OSKut/blob/main/notebooks/OSKut_how1.ipynb)

Handling Cross- and Out-of-Domain Samples in Thai Word Segmentation (ACL 2021 Findings) <br>
Stacked Ensemble Framework and DeepCut as Baseline model<br>

## Read more:
- Paper: TBD
- Related Work (EMNLP2020): [Domain Adaptation of Thai Word Segmentation Models using Stacked Ensemble](https://www.aclweb.org/anthology/2020.emnlp-main.315/)
- Blog: [Domain Adaptation กับตัวตัดคำ มันดีย์จริงๆ](https://medium.com/@pingloaf)

## Citation
```
@inproceedings{limkonchotiwat-etal-2021-handling,
    title = "Handling Cross- and Out-of-Domain Samples in {T}hai Word Segmentation",
    author = "Limkonchotiwat, Peerat  and
      Phatthiyaphaibun, Wannaphong  and
      Sarwar, Raheem  and
      Chuangsuwanich, Ekapol  and
      Nutanong, Sarana",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.86",
    doi = "10.18653/v1/2021.findings-acl.86",
    pages = "1003--1016",
}
```

## Install
> pip install OSKut

## How To use
### Requirements
- python >= 3.6
- tensorflow >= 2.0

## Example
- Example files are on [OSKut Example notebook](https://github.com/mrpeerat/OSKut/blob/main/notebooks/OSKut_how1.ipynb)
- [Try it on Colab](https://colab.research.google.com/github/mrpeerat/OSKut/blob/main/notebooks/OSKut_how1.ipynb)
### Load Engine & Engine Mode
- ws, tnhc, and BEST !!
  - ws: The model trained on Wisesight-1000 and test on Wisesight-160
  - ws-augment-60p: The model trained on Wisesight-1000 augmented with top-60% entropy
  - tnhc: The model trained on TNHC (80:20 train&test split with random seed 42)
  - BEST: The model trained on BEST-2010 Corpus (NECTEC)
  - SCADS: The model trained on VISTEC-TP-TH-2021 Corpus (VISTEC)
  ```python
  oskut.load_model(engine='ws')
  # OR
  oskut.load_model(engine='ws-augment-60p')
  # OR
  oskut.load_model(engine='tnhc')
  # OR
  oskut.load_model(engine='best')
  # OR
  oskut.load_model(engine='scads')
  # OR
  ```
- tl-deepcut-XXXX
  - We also provide transfer learning of deepcut on 'Wisesight' as tl-deepcut-ws, 'TNHC' as tl-deepcut-tnhc, and 'LST20' as tl-deepcut-lst20
  ```python
  oskut.load_model(engine='tl-deepcut-ws')
  # OR
  oskut.load_model(engine='tl-deepcut-tnhc')
  ```
- deepcut
  - We also provide the original deepcut
  ```python
  oskut.load_model(engine='deepcut')
  ```
### Segment Example
You need to read the paper to understand why we have $k$ value!
- Tokenize with default k-value
  ```python
  oskut.load_model(engine='ws')
  print(oskut.OSKut(['เบียร์ยูไม่อร่อยสัดๆๆๆๆๆฟๆ']))
  print(oskut.OSKut('เบียร์ยูไม่อร่อยสัดๆๆๆๆๆฟๆ'))
  
  ['เบียร์', 'ยู', 'ไม่', 'อร่อย', 'สัด', 'ๆ', 'ๆ', 'ๆ', 'ๆ', 'ๆฟ', 'ๆ']
  ['เบียร์', 'ยู', 'ไม่', 'อร่อย', 'สัด', 'ๆ', 'ๆ', 'ๆ', 'ๆ', 'ๆฟ', 'ๆ']
  ```
- Tokenize with a various k-value
  ```python
  oskut.load_model(engine='ws')
  print(oskut.OSKut('เบียร์ยูไม่อร่อยสัดๆๆๆๆๆฟๆ',k=5)) # refine only 5% of character number
  print(oskut.OSKut('เบียร์ยูไม่อร่อยสัดๆๆๆๆๆฟๆ',k=100)) # refine 100% of character number
  
  ['เบียร์', 'ยู', 'ไม่', 'อร่อย', 'สัด', 'ๆ', 'ๆ', 'ๆ', 'ๆ', 'ๆฟๆ']
  ['เบียร์', 'ยู', 'ไม่', 'อร่อย', 'สัด', 'ๆ', 'ๆ', 'ๆ', 'ๆ', 'ๆฟ', 'ๆ']
  ```
  
## New datasets!!
VISTEC-TP-TH-2021 (VISTEC), which consists of 49,997 text samples from Twitter (2017-2019). \
VISTEC corpus contains 49,997 sentences with 3.39M words where the collection was manually annotated by linguists on four tasks, namely word segmentation, misspelling detection and correction, and named entity recognition. \
For more information and download [click here](https://github.com/mrpeerat/OSKut/tree/main/VISTEC-TP-TH-2021)

## Performance
### Model
<img src="https://user-images.githubusercontent.com/21156980/117925237-24f10500-b321-11eb-8e69-8efee577e1d7.png" width="600"/>

### Without Data Augmentation
<img src="https://user-images.githubusercontent.com/21156980/117925463-75686280-b321-11eb-8e39-fcdae3c569ea.png" width="600"/>

### With Data Augmentation
<img src="https://user-images.githubusercontent.com/21156980/117925342-4d78ff00-b321-11eb-80fa-59d71ce46a5a.png" width="600"/>
<img src="https://user-images.githubusercontent.com/21156980/117925347-4f42c280-b321-11eb-86a3-475b876b8851.png" width="600"/>


Thank you many code from

- [Deepcut](https://github.com/rkcosmos/deepcut) (Baseline Model) : We used some of code from Deepcut to perform transfer learning 

