<h1 style="text-align:center;">FOFE in Adapter</h1>

The FOFE (Fixed-Size Ordered Frequent-pattern based Embedding) layer is a type of neural network layer used in NLP tasks. It is used for learning word representations by combining the information from frequent patterns and their order in the data.

The FOFE layer is designed to take as input a bag-of-words representation of a text sequence, where the words are represented by their frequency in the sequence. The layer then generates a fixed-size vector representation for each word, capturing the information about the frequent patterns and the order of the words in the sequence.

Adapters are usually implemented as feed-forward neural networks with a small number of hidden layers, and they can be added to various parts of a pre-trained model.

Overall, the FOFE layer and adapter layer can be combined together, and can be applied in various NLP applications to improve the quality of text representation and modeling. In FOFE Adapter, the inputs embeddings pass through a reduction layer, an FOFE layer and an expand layer. There are 4 different window sizes in FOFE layer. The model aims to learn the weights distribution of each window size.

For example, we can set **window_size** as 3, and **input_len** as 8. The **FOFE matrix** is shown below.

![图片1.png](https://img1.imgtp.com/2023/02/13/SgiSi4yp.png)


## run the scripts:

bash run_glue_adapter.bash

