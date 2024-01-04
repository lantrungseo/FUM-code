## Paper

Attention pooling: https://arxiv.org/pdf/1602.03609.pdf
Fastformer: https://arxiv.org/pdf/2108.09084.pdf

## How to run 

### Google Colab

In the Google Drive root folder

- Create /Projects/MIND-FUM/ folder
- Create /Dataset/MIND/_data_ folder
- Create /Dataset/MIND/_embeddings_ folder
- Download and extract MIND dataset to /Dataset/MIND/_data_ folder (https://msnews.github.io/)
- Download GloVe embeddings (840B.300d)  to /Dataset/MIND/_embeddings_ folder (https://nlp.stanford.edu/projects/glove/)
- Upload `Generator.py, Hypers.py, Models.py, Utils.py, Preprocessing.py` to the `/Projects/MIND-FUM/`
- Run the `FUM-train.ipynb` notebook to train the model in Google Colab
- Run the `FUM_code_eval.ipynb` notebook to evaluate the model in Google Colab
- Go to Google Drive and use the `/Dataset/MIND/prediction.txt` to submit to the competition