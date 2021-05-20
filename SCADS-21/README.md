# New dataset "SCADS-21"

Co-operate VISTEC x CMU (XXXXX)

# Detail
## Motivation and detail
Due to social media data being underrepresented and difficult, it challenging to improve the performance of models with only 997 training sentences.\
Most TWS models (deepcut,attacut) performed under 82% in out-of-domain social media scenarios (Wisesight).\
To address this problem, we introduce a new dataset called 'Scalable Data Systems 2021' (SCADS-21), which consists of 49,997 text samples from Twitter.\
SCADS-21 corpus contains 49,997 sentences with 3.39M words which the collection was manually annotated by linguists on four tasks namely word segmentation, misspelling detection \& correction, and named entity recognition.\
In the data collection process, we focused on the longest sentences to create a more challenging dataset.\
The Out-of-Vocabulary rate on the test set is 13.65.\
We followed LST20's work for the word and named entity tasks.\
We also included new guidelines about word editing criteria for misspelling words such as words used on the internet (Netspeak), transliterated loanwords, abbreviations, and shortened words, by using the Royal Institute Thai dictionary.\
We compared our dataset to the biggest [Thai social media dictionary](https://github.com/Knight-H/thai-lm) and found 79K words that did not appear in the dictionary.\
## Task inside
- Word segmentation
- Misspell detection & correction
- Named-entity boundary
