# [PlaceHolder]

## Code Directory
```
.
|-- data                                        : code related to data processing
|   |-- build_pretrain_data.py                      : build pre-training data
|   |-- const.py                                    : consists of some constant variables
|   |-- dataset.py                                  : multiple Pytorch Dataset objects
|   |-- merge_blocks.py	                            : merge data blocks (to prerocess on multiple machines)
|   |-- split_blocks.py                             : split data blocks	(to prerocess on multiple machines)
|   `-- utils.py                                    : some utility functions
|-- downstream                                  : code for fine-tuing on downstream tasks
|   |-- eval                                        : evaluation scripts
|   |   |-- squad_eval_v1.py                            : official evaluation script for SQuAD V1
|   |   `-- squad_eval_v2.py                            : official evaluation script for SQuAD V2
|   `-- run                                         : fine-tuning scripts
|       |-- finetune_seq_classification.py              : fine-tuning of sequence classification tasks
|       `-- finetune_squad.py                           : fine-tuning of SQuAD
|-- model                                       : model code
|   |-- BiBeam.py                                   : Bi-directional Beam Search
|   |-- __init__.py                                     : Python init py
|   |-- config.py                                       : model config
|   |-- decoder.py                                      : unified decoder
|   |-- model.py                                        : model
|   `-- tokenizer.py                                    : tokenizer
|-- scripts                                     : various scripts to execute directly
|   |-- build_pretrain_data.sh                      : build pre-training data
|   |-- pred.slurm                                  : prediction
|   |-- preprocess.sh                               : preprocessing for downstream tasks
|   |-- pretrain.sh                                 : pre-train
|   |-- run.sh                                      : test running script
|   `-- run_squad.sh                                : fine-tune squad
|-- tests                                       : test code directory for development
|   |-- rm_invalid_lines.py
|   |-- test_file_dataset.py
|   |-- test_file_pointer.py
|   |-- test_hostname.py
|   |-- test_model.py
|   |-- test_pretrain_data_step1.py
|   |-- test_reverse_data.py
|   |-- test_squad_dataset.py
|   `-- test_tokenizer.py
|-- utils                                       : some utility functions
|   |-- custom_logger.py                            : customized logger
|   |-- launch.py                                   : customized file-interaction launcher for multi-machine multi GPU
|   |-- preprocessing.py                            : preprocessing for downstream tasks
|   |-- trainer.py                                  : a variety of trainers
|   `-- training_args.py                            : training arguments supporting file-interaction
|-- README.md
|-- __init__.py
|-- pretrain.py                                 : pre-train
`-- run.py                                      : test running
```



## Procedure

### Pre-train

#### overview

* construct pre-training data from orgin BookCorpus and English Wikipedia
* pre-train

#### construct pre-training data

**NOTE: the processed pre-training data is located at:**

```
/mnt/lustre/sjtu/home/dm311/remote/pretrain/data/pretrain
```

* Preprocessing for Pretrained data

  * **The preprocessed data is located at**

    ```
    bookcorpus: /mnt/lustre/sjtu/home/dm311/remote/PLUTO/data/pretrain/bookcorpus
    wiki: /mnt/lustre/sjtu/home/dm311/remote/PLUTO/data/pretrain/wiki
    ```

  * If you do not want to use our processed data, you can build it yourself according to following process.

    **NOTE: this part of code is not uploaded now !!!**

    * Get the original dataset: We use [BookCorpus](https://t.co/J3EaSEgwW0?amp=1) and [EnglishWikipedia](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2).

    * Preprocess each dataset into documents seperated by an empty line

      ```shell
      # For BookCorpus, we regard each chapter as a document.
      ./scripts/bookcorpus_doc_partition.sh <dataset-dir> <save-dir> <book-num-for-each-process>
      
      # example
      ./scripts/bookcorpus_doc_partition.sh ../data/BookCorpus ../data/pretrain/bookcorpus_doc 10
      
      # ------------------------------------------------------------------------ #
      
      # For EnglishWikipedia, use this [repo](https://github.com/attardi/wikiextractor) (used in **BERT**).
      # get the main contents first (remove tables, urls and so on)
      python -m wikiextractor.WikiExtractor <wiki-xml-file> --output <output-dir>
      
      # remove headers and split paragraphs of each document into tokenized sentences
      ./scripts/wiki_doc_partition.sh <extracted-data-dir> <save-dir>
      ```

      

    * Tokenize each document into sentence-by-sentence format

      ``` shell
      ./scripts/to_sent_line_by_line.sh <doc-data-dir> <save-dir> <file-num-for-each-process>
      
      # example
      ./scripts/to_sent_line_by_line.sh ../data/pretrain/wiki_doc ../data/pretrain/wiki 2
      ```

    * Split each dataset into blocks

      ``` shell
      ./scripts/split.sh <line-by-line-sentence-dir> <block-to-save-dir> <block_size> empty
      
      # i.e. 
      # EnglishWikipedia
      ./scripts/split.sh ../data/pretrain/wiki ../data/pretrain/train 102400 empty
      ```

* Introduce noises and build pre-training data

  ```shell
  # construct pre-training data for each block
  # NOTE: config ./scripts/build_pretrain_data.sh properly !!!
  sbatch scripts/build_pretrain.sh
  
  # merge blocks
  # NOTE: Also, config properly !!!
  sbatch scripts/merge.sh
  
  # Build pre-training data index file && build segment partition file
  # NOTE: also run <build_pretrain_data.sh>, but requires proper config !!!
  ```

#### pre-train our model

```shell
	# config properly
	sbatch scripts/pretrain.sh
```



### Fine-tune

