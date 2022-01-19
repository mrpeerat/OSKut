# New dataset "VISTEC-TP-TH-21"
The largest social media domain datasets for Thai text processing (word segmentation, misspell correction and detection, and named-entity boundary) called "VISTEC-TP-TH-2021" or **VISTEC-2021**\
Co-operate between Vidyasirimedhi Institute of Science and Technology, Thailand and Chiang Mai University, Thailand.

# Motivation and detail
Due to social media data being underrepresented and difficult, it is challenging to improve the performance of models with only 997 training sentences. \
Most TWS models performed under 82\% in out-of-domain social media scenarios (Wisesight).\
To address this problem, we introduce a new dataset called VISTEC-TP-TH-2021 (VISTEC), which consists of 49,997 text samples from Twitter (2017-2019).\
VISTEC corpus contains 49,997 sentences with 3.39M words where the collection was manually annotated by linguists on four tasks, namely word segmentation, misspelling detection and correction, and named entity recognition.\
In the data collection process, we focused on the longest sentences to create a more challenging dataset due to the fact that long sentence made the model's performance decrease significantly comparing with short sentence in the same domain. The Out-of-Vocabulary rate on the test set is 13.65%.

We followed LST20's work for the word and named entity tasks annotation guideline.\
We also included new guidelines about word editing criteria for misspelt words such as words used on the internet (Netspeak), transliterated loanwords, abbreviations, and shortened words, by using the Royal Institute Thai dictionary.\
We compared our dataset to the biggest [Thai social media dictionary](https://github.com/Knight-H/thai-lm) and found 79K words that did not appear in the dictionary.

## Tasks inside
- Word segmentation: speacial character:```|```
- Misspell detection & correction: tag: ```<msp>``` for detection and tag: ```<msp='XX'>XXX</msp>``` for correction
- Named-entity boundary: tag: ```<ne>```

## Criterial
- https://github.com/mrpeerat/OSKut/blob/main/VISTEC-TP-TH-2021/Criteria.pdf

## Download word segmentation processed data
- Link: https://drive.google.com/drive/folders/1-9D1E0iSvPSBLawSHeJiqUFhKHOhQ4-j?usp=sharing

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

## Developer and Annotator from Chiang Mai University
Corpus by \
Saroj Buaphanngam สาโรจน์ บัวพันธุ์งาม \
Pornwipa Chaisomkhun พรวิภา ไชยสมคุณ \
Chayanin Boonsongsak ชญานิน บุญส่งศักดิ์

Supported by \
Titipat Sukhvibul ธิติพัฒน์ สุขวิบูลย์\
Juggapong Natwichai จักรพงศ์ นาทวิชัย

## License

CC-BY-SA 3.0
