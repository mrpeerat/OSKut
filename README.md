# SKut (StacKed cut with Filter and Refine framework for Word Segmentation) 
Handling Cross- and Out-of-Domain Samples in Thai Word Segmentation (ACL 2020 Findings) <br>
Stacked Ensemble Framework and DeepCut as Baseline model<br>

## Read more:
- Paper: tbd
- Previous Paper: [Domain Adaptation of Thai Word Segmentation Models using Stacked Ensemble](https://www.aclweb.org/anthology/2020.emnlp-main.315/)
- Blog: [Domain Adaptation กับตัวตัดคำ มันดีย์จริงๆ](https://medium.com/@pingloaf)

## Citation
```
@inproceedings{limkonchotiwat-etal-2020-domain,
    title = "Domain Adaptation of {T}hai Word Segmentation Models using Stacked Ensemble",
    author = "Limkonchotiwat, Peerat  and
      Phatthiyaphaibun, Wannaphong  and
      Sarwar, Raheem  and
      Chuangsuwanich, Ekapol  and
      Nutanong, Sarana",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.315",
}
```

## Install
> pip install xxxxxx

## How To use
### Requirements
- python >= 3.6
- python-crfsuite >= 0.9.7
- pyahocorasick == 1.4.0

## Example
- Example files are on [SEFR Example notebook](https://github.com/mrpeerat/SEFR_CUT/blob/master/Notebooks/1.SEFR_CUT%20example.ipynb)
- [Try it on Colab](https://colab.research.google.com/drive/1xA2rzYVnVWwxy6oFkISiG63x-5u1gwa1?usp=sharing)
### Load Engine & Engine Mode
- ws1000, tnhc, and BEST !!
  - ws1000: The model trained on Wisesight-1000 and test on Wisesight-160
  - tnhc: The model trained on TNHC (80:20 train&test split with random seed 42)
  - BEST: The model trained on BEST-2010 Corpus (NECTEC)
  ```python
  sefr_cut.load_model(engine='ws1000')
  # OR
  sefr_cut.load_model(engine='tnhc')
  # OR
  sefr_cut.load_model(engine='best')
  ```
- tl-deepcut-XXXX
  - We also provide transfer learning of deepcut on 'Wisesight' as tl-deepcut-ws1000 and 'TNHC' as tl-deepcut-tnhc
  ```python
  sefr_cut.load_model(engine='tl-deepcut-ws1000')
  # OR
  sefr_cut.load_model(engine='tl-deepcut-tnhc')
  ```
- deepcut
  - We also provide the original deepcut
  ```python
  sefr_cut.load_model(engine='deepcut')
  ```
### Segment Example
You need to read the paper to understand why we have $k$ value!
- Tokenize with default k-value
  ```python
  sefr_cut.load_model(engine='ws1000')
  print(sefr_cut.tokenize(['สวัสดีประเทศไทย','ลุงตู่สู้ๆ']))
  print(sefr_cut.tokenize(['สวัสดีประเทศไทย']))
  print(sefr_cut.tokenize('สวัสดีประเทศไทย'))
  
  [['สวัสดี', 'ประเทศ', 'ไทย'], ['ลุง', 'ตู่', 'สู้', 'ๆ']]
  [['สวัสดี', 'ประเทศ', 'ไทย']]
  [['สวัสดี', 'ประเทศ', 'ไทย']]
  ```
- Tokenize with a various k-value
  ```python
  sefr_cut.load_model(engine='ws1000')
  print(sefr_cut.tokenize(['สวัสดีประเทศไทย','ลุงตู่สู้ๆ'],k=5)) # refine only 5% of character number
  print(sefr_cut.tokenize(['สวัสดีประเทศไทย','ลุงตู่สู้ๆ'],k=100)) # refine 100% of character number
  
  [['สวัสดี', 'ประเทศไทย'], ['ลุงตู่', 'สู้', 'ๆ']]
  [['สวัสดี', 'ประเทศ', 'ไทย'], ['ลุง', 'ตู่', 'สู้', 'ๆ']]
  ```

## Evaluation
- We also provide Character & Word Evaluation by call function ```evaluation()``` 
  - For example
  ```python
  answer = 'สวัสดี|ประเทศไทย'
  pred = 'สวัสดี|ประเทศ|ไทย'
  char_score,word_score = sefr_cut.evaluation(answer,pred)
  print(f'Word Score: {word_score} Char Score: {char_score}')

  Word Score: 0.4 Char Score: 0.8
  
  answer = ['สวัสดี|ประเทศไทย']
  pred = ['สวัสดี|ประเทศ|ไทย']
  char_score,word_score = sefr_cut.evaluation(answer,pred)
  print(f'Word Score: {word_score} Char Score: {char_score}')

  Word Score: 0.4 Char Score: 0.8
  
  
  answer = [['สวัสดี|'],['ประเทศไทย']]
  pred = [['สวัสดี|'],['ประเทศ|ไทย']]
  char_score,word_score = sefr_cut.evaluation(answer,pred)
  print(f'Word Score: {word_score} Char Score: {char_score}')
  
  Word Score: 0.4 Char Score: 0.8
  ```

## Performance
<img src="https://user-images.githubusercontent.com/21156980/117923338-0dfce380-b31e-11eb-96f2-e61ccd6b9f20.png" width="600" height="386" />

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
  - Link:[HERE](https://github.com/mrpeerat/SEFR_CUT/blob/master/Notebooks/3.Stacked%20Model%20Example.ipynb)
  ### Use your trained model?
  - Just move your model inside 'Notebooks/model/' to 'seft_cut/model/' and call model in one line.
  ```python
  SEFR_CUT.load_model(engine='my_model')
  ```

Thank you many code from

- [Deepcut](https://github.com/rkcosmos/deepcut) (Baseline Model) : We used some of code from Deepcut to perform transfer learning 
- [@bact](https://github.com/bact) (CRF training code) : We used some from https://github.com/bact/nlp-thai


