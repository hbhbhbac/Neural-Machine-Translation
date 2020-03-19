# Neural Machine Translation with Sequence to Sequence RNN and Deeplearning4j

The task of machine translation consists of reading text in one language and generating text in another language while keeping the same meaning as that of the reading text. 
When neural networks are used to perform this task, we talk about neural machine translation (NMT). I present here a Deeplearning4j-based implementation of NMT with Sequence to Sequence Recurrent Neural Network (RNN). 
This implementation is based on the exemple that models the sequence to sequence RNNs used for the addition operation and which can be found [here](https://github.com/eclipse/deeplearning4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/seq2seq).
The NMT will attempt to perforn an English to French translation experiment.

The dataset user to train this NMT can be freely downloaded at [http://www.manythings.org/anki/](http://www.manythings.org/anki/). 
It contains commonly used sententenses in both languages. The data of this dataset have been transfered to csv files and arranged in 
lines with each composed of a sentence in english language and its corresponding translation in french language, both seperated by a comma. (see the following figure)

![dataset_arranged](https://user-images.githubusercontent.com/1300982/76970927-d7e7ed80-692c-11ea-8345-ba5630490e83.png)

Furthermore, these data have been splitted out so as to get the training and test data in to two seperated csv files.

Training the network with few training is already producing very promising results on test data.

![results](https://user-images.githubusercontent.com/1300982/77025761-3dba9080-6992-11ea-9c65-d78aeca40d90.png)

![res](https://user-images.githubusercontent.com/1300982/77025920-ab66bc80-6992-11ea-8d41-bf47449043aa.png)

