# NLP Assignment: Aspect-Term Polarity Classification in Sentiment Analysis

The goal of this assignment is to implement a classifier that predicts opinion polarities (positive, negative or neutral) for given aspect terms in sentences. The classifier takes as input 3 elements: a sentence, an aspect term occurring in the sentence, and its aspect category. For each input triple, it produces a polarity label: positive, negative or neutral.

## Authors
The project is done by  [Antoine Cloute](https://github.com/AntAI-Git), [Annabelle Luo](https://github.com/annabelleluo), [Konstantina Tsilifoni](https://github.com/KonstantinaTsili).

## Model Description
### 1. Data Augementation
To improve the performance of the model and avoid the potential limitations resulting from the limited training dataset, the original training data is augmented using an adjusted EDA technique: synonym replacement and swap target words while considering the specifics of ABSA.
The logic of the data augmentation methods has been derived from the two following papers:
-	 "Data Augmentation in a Hybrid Approach for Aspect-Based Sentiment Analysis" by Tomas Liesting, Flavius Frasincar, Maria Mihaela Trusca (https://arxiv.org/abs/2103.15912)
-	"Improving Short Text Classification Through Global Augmentation Methods" by Vukosi Marivate & Tshephisho Sefara (https://link.springer.com/chapter/10.1007/978-3-030-57321-8_21)

### 2. Model
Our model is based on a pre-trained BERT transformers model that encodes the input sentences. The pretrained BERT model is based on a restaurant specific extended version of the initial BERT presented by Devlin et al. 2018. Indeed, the original BERT model has been post-trained in a self-supervised way (NSP and MLM) on restaurant reviews coming from Yelp (Xu et al. published in NAACL 2019). We then use the encoded sentences as input to a classification model to predict the polarities. For the aspect-based sentiment classification (ABS), we use a local context focus mechanism model proposed by Zeng et al. , which internally employs a BERT self-attention with multiple attention heads.

### 3. Results
Our model achieved the following results over 5 runs:
- Dev accs: [88.56, 88.83, 86.97, 86.97, 86.44]
- Test accs: [-1, -1, -1, -1, -1]
- Mean Dev Acc.: 87.55 (0.96)
- Mean Test Acc.: -1.00 (0.00)
- Exec time: 2990.76 s. ( 598 per run )

## Instruction for Running
To train and evaluate the model, navigate to the "src" folder, and run the "tester.py" script using the command:
```
“python tester.py” 
```
There are 2 available optional arguments:
-	n_runs: number of runs you want to execute. The default is 5.
-	gpu: The ID of the GPU on which you want to run the model.

