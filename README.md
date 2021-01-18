# ![quati](quati-logo.png) quati
Simple and modular library for document classification and sequence tagging.

### Usage

First, create a virtualenv using your favorite tool, and then install all dependencies using:

```bash
pip3 install -r requirements.txt
```

Finally, install it as a library with:
```bash
python3 setup.py install
```

You can see a complete help message by running:
```bash
python3 -m quati --help
```

### Data

1. Download the datasets with the script `download_datasets.sh` (1.6G). 
Yelp dataset should be downloaded separately (6G).

2. Then run the script `bash generate_dataset_partitions.sh` to create train/dev/test partitions for AgNews, IMDB and Yelp.

3. If you want to use GloVe embeddings (as in our paper), you have two options:

    a) Use the script `download_glove_embeddings.sh` to download all embedding vectors. 
And, if you want to use only the embeddings for a particular corpus, i.e.,
restrict the embeddings vocabulary to the corpus vocabulary for all downloaded corpus,
use the script `scripts/reduce_embeddings_model_for_all_corpus.sh`. 

    b) Download the already restricted-to-vocab glove embeddings with the script `download_restricted_glove_embeddings.sh`.
    [UPDATE: dropbox link is not working anymore.]

4. (optional) Create folders for saved models and attentions:
    ```sh
    mkdir -p data/saved-models
    mkdir -p data/attentions
    ```

Here is how your data folder will be organized:
```
data/
├── corpus
│   ├── agnews
│   ├── imdb
│   ├── snli
│   └── sst
│   └── yelp
├── embs
│   └── glove
├── saved-models
├── attentions
```


### How to run

Use the command `train` to train a classifier. For example, see `experiments/train_sst.sh`.
Take a look in the `experiments` folder for more examples.


### Continuous attention

See `train_imdb_discrete.sh`, `train_imdb_continuous.sh`, `train_imdb_discrete_and_continuous.sh` in the experiments folder.
Run them to replicate the results reported in our paper [Sparse and Continuous Attention Mechanisms](https://arxiv.org/abs/2006.07214) [1].

[1] André F. T. Martins, António Farinhas, Marcos Treviso, Vlad Niculae, Pedro M. Q. Aguiar, and Mário A. T. Figueiredo. [Sparse and Continuous Attention Mechanisms](https://arxiv.org/abs/2006.07214). NeurIPS 2020.


### License

MIT.


### Name
[Nasua nasua](https://en.wikipedia.org/wiki/South_American_coati) (aka quati in Brazil) is a member of the raccoon family.
Logo from [here](https://www.vectorstock.com/royalty-free-vector/flat-style-of-nasua-vector-13611463).
