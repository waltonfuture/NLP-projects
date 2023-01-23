# NLP-projects

There are several projects about NLP and data mining.


## Diabetes Genetic Risk Prediction

[Diabetes Genetic Risk Prediction](https://challenge.xfyun.cn/topic/info?type=diabetes&ch=ds22-dw-wd01) is a data mining competition held by XunFei. It aims to build 
a diabetes genetic risk prediction model from the training data set, and then predict whether individuals in the test data set have diabetes. 

I make use of three machine learning model for this task: Logistic Regression, Decision Tree, and Lightgbm.

## Text Classification and Query Question Answering Public Data Based on Paper Abstracts

[Text Classification and Query Question Answering Public Data Based on Paper Abstracts](https://challenge.xfyun.cn/topic/info?type=abstract&ch=ds22-dw-wd07) is a NLP classification competition
held by XunFei. A model is needed to classify the papers by understanding the abstracts and other information of them. 

Baseline model is a SGDClassifier. I use bert-base-uncased pretrained model to improve the performance.

## FOFE Adapter

Fifixedsize ordinally-forgetting encoding (FOFE) can almost uniquely encode any variable-length sequence of words into a fifixed-size representation. It can model the word order in a sequence using a simple ordinally-forgetting mechanism according to the positions of words. FOFELayerWindows are added to the BERT encoder to help the pretrained model better understand tokens' relationship based on GLUE dataset.
