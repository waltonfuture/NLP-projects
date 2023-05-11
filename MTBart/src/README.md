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
|       
|-- model                                       : model code
|   |-- BiBeam.py                                   : Bi-directional Beam Search
|   |-- __init__.py                                     : Python init py
|   |-- config.py                                       : model config
|   |-- decoder.py                                      : unified decoder
|   |-- model.py                                        : model
|   `-- tokenizer.py                                    : tokenizer
|-- scripts                                     : various scripts to execute directly
|   
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



![mtbart1.png](https://img1.imgtp.com/2023/05/11/yKN6EafU.png)

![mtbart2.png](https://img1.imgtp.com/2023/05/11/xHbWaPho.png)

