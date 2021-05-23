# <span style="color:orange;">OS</span>Kut (Out-of-domain StacKed cut with Filter and Refine framework for Word Segmentation) 
Handling Cross- and Out-of-Domain Samples in Thai Word Segmentation (ACL 2020 Findings) <br>
Stacked Ensemble Framework and DeepCut as Baseline model<br>

## Read more:
- Paper: TBD
- Related Work: [Domain Adaptation of Thai Word Segmentation Models using Stacked Ensemble](https://www.aclweb.org/anthology/2020.emnlp-main.315/)
- Blog: [Domain Adaptation กับตัวตัดคำ มันดีย์จริงๆ](https://medium.com/@pingloaf)

## Citation
```
TBD
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
- ws1000, tnhc, and BEST !!
  - ws1000: The model trained on Wisesight-1000 and test on Wisesight-160
  - tnhc: The model trained on TNHC (80:20 train&test split with random seed 42)
  - BEST: The model trained on BEST-2010 Corpus (NECTEC)
  - SCADS: The model trained on SCADS-21 Corpus (VISTEC)
  ```python
  oskut.load_model(engine='ws1000')
  # OR
  oskut.load_model(engine='tnhc')
  # OR
  oskut.load_model(engine='best')
  ```
- tl-deepcut-XXXX
  - We also provide transfer learning of deepcut on 'Wisesight' as tl-deepcut-ws1000 and 'TNHC' as tl-deepcut-tnhc
  ```python
  oskut.load_model(engine='tl-deepcut-ws1000')
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
  oskut.load_model(engine='ws1000')
  print(oskut.OSKut(['สวัสดีประเทศไทย','ลุงตู่สู้ๆ']))
  print(oskut.OSKut(['สวัสดีประเทศไทย']))
  print(oskut.OSKut('สวัสดีประเทศไทย'))
  
  [['สวัสดี', 'ประเทศ', 'ไทย'], ['ลุง', 'ตู่', 'สู้', 'ๆ']]
  [['สวัสดี', 'ประเทศ', 'ไทย']]
  [['สวัสดี', 'ประเทศ', 'ไทย']]
  ```
- Tokenize with a various k-value
  ```python
  oskut.load_model(engine='ws1000')
  print(oskut.OSKut(['สวัสดีประเทศไทย','ลุงตู่สู้ๆ'],k=5)) # refine only 5% of character number
  print(oskut.OSKut(['สวัสดีประเทศไทย','ลุงตู่สู้ๆ'],k=100)) # refine 100% of character number
  
  [['สวัสดี', 'ประเทศไทย'], ['ลุงตู่', 'สู้', 'ๆ']]
  [['สวัสดี', 'ประเทศ', 'ไทย'], ['ลุง', 'ตู่', 'สู้', 'ๆ']]
  ```
## New datasets!!
```
TBD
```

## Performance
### Model
<img src="https://user-images.githubusercontent.com/21156980/117925237-24f10500-b321-11eb-8e69-8efee577e1d7.png" width="600"/>

### Without Data Augmentation
<img src="https://user-images.githubusercontent.com/21156980/117925463-75686280-b321-11eb-8e39-fcdae3c569ea.png" width="600"/>

### With Data Augmentation
<img src="https://user-images.githubusercontent.com/21156980/117925342-4d78ff00-b321-11eb-80fa-59d71ce46a5a.png" width="600"/>
<img src="https://user-images.githubusercontent.com/21156980/117925347-4f42c280-b321-11eb-86a3-475b876b8851.png" width="600"/>

## How to re-train the model?
- You can re-train the model. The example is in the folder [Notebooks](https://github.com/mrpeerat/SEFR_CUT/tree/master/Notebooks) We provided everything for you!!
  ### Re-train Model
  - You can run the notebook file #2, the corpus inside 'Notebooks/corpus/' is Wisesight-1000, you can try with BEST, TNHC, and LST20 !
  - Rename variable name: ```CRF_model_name``` 
  - Link:[HERE](https://github.com/mrpeerat/SEFR_CUT/blob/master/Notebooks/2.Train_DS_model.ipynb)
  ### Filter and Refine Example
  - Set variable name ```CRF_model_name``` same as file#2 
  - If you want to know why we use filter-and-refine, you can try to uncomment 3 lines in ```score_()``` function
  ```
  #answer = scoring_function(y_true,cp.deepcopy(y_pred),entropy_index_og)
  #f1_hypothesis.append(eval_function(y_true,answer))
  #ax.plot(range(start,K_num,step),f1_hypothesis,c="r",marker='o',label='Best case')
  ```
  - Link: [HERE](https://github.com/mrpeerat/SEFR_CUT/blob/master/Notebooks/3.Stacked%20Model%20Example.ipynb)
  ### Use your trained model?
  - Just move your model inside 'Notebooks/model/' to 'oskut/model/' and call model in one line.
  ```python
  SEFR_CUT.load_model(engine='my_model')
  ```

Thank you many code from

- [Deepcut](https://github.com/rkcosmos/deepcut) (Baseline Model) : We used some of code from Deepcut to perform transfer learning 
- [Arthit Suriyawongkul ](https://github.com/bact) (CRF training code) : We used some from https://github.com/bact/nlp-thai


