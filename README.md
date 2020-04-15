# Neural Machine Translation with Sequence to Sequence RNN and Deeplearning4j

The task of machine translation consists of reading text in one language and generating text in another language while keeping the same meaning as that of the reading text. 
When neural networks are used to perform this task, we talk about neural machine translation (NMT). I present here a Deeplearning4j-based implementation of NMT with Sequence to Sequence Recurrent Neural Network (RNN). 
This implementation is based on the exemple that models the sequence to sequence RNNs used for the addition operation and which can be found [here](https://github.com/eclipse/deeplearning4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/seq2seq).
The NMT will attempt to perforn an English to French translation experiment.

Our neural machine translation will be implemented at character level where the next character is predicted given the previous sequence of characters.

Because of the fact that this NMT is dealing with two languages (English and French), we adopted a neural architecture with two LSTM layers which  is known as the encoder-decoder RNN structure. In this architecture, one LSTM layer is used for the  original language (English) and the other LSTM layer is used for the target language (French).

The dataset used to train this NMT can be freely downloaded at [http://www.manythings.org/anki/](http://www.manythings.org/anki/). 
It contains commonly used sententenses in both languages. The data of this dataset have been transfered to csv files and arranged in 
lines with each composed of a sentence in english language and its corresponding translation in french language, both seperated by a comma. (see the following figure)

![dataset_arranged](https://user-images.githubusercontent.com/1300982/76970927-d7e7ed80-692c-11ea-8345-ba5630490e83.png)

Furthermore, these data have been splitted out so as to get the training and test data in to two seperated csv files.

Training the network with few training data is already producing very promising results on test data, as shown in the following two figures. The first figure depicts the Model Score vs. Iteration, which is the visualization of the loss plot. It allows us to visualize the loss of the network against each iteration and check the model performance.

![shart](https://user-images.githubusercontent.com/1300982/78967582-ca242300-7afa-11ea-8b5c-678db76f8d01.png)

The second figure presents the predictions of the model when fed with the English sentences from the test data. I have highlighted the lines with sentences in French language that were badly or partially predicted (translated).

![results](https://user-images.githubusercontent.com/1300982/79002776-14cc8c00-7b49-11ea-895e-a04e2b06c10f.png)


