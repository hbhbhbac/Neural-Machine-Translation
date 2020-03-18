# Neural Machine Translation with Sequence to Sequence RNN and Deeplearning4j

The task of machine translation consists of reading text in one language and generating text in another language while keeping the same meaning as that of the reading text. 
When neural networks are used to perform this task, we talk about neural machine translation (NMT). I present here a Deeplearning4j-based implementation of NMT with Sequence to Sequence Recurrent Neural Network (RNN). 
This implementation is based on the exemple that models the sequence to sequence RNNs used for the addition operation and which can be found [here](https://github.com/eclipse/deeplearning4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/seq2seq).
The NMT will attempt to perforn an English to French translation experiment.

The dataset user to train this NMT can be freely downloaded at [http://www.manythings.org/anki/](http://www.manythings.org/anki/). 
It contains commonly used sententenses in both languages. The data of this dataset have been transfered to a csv file and arranged in 
lines with each composed of a sentence in english language and its corresponding translation in french language, both seperated by a comma
