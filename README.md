# unmasker
Unsuccesful approach to unmasking words using AI model

# Experimental Design 
Used various approaches, first starting with building a linear model with LSTM layers to process definition and masked words separately adn then concatenate. Then, I added a BERT layer to process definition and take its hidden state. Both approaches had too much loss and did not grasp the meanings of the inputs correctly. The code can be found under previous_iterations. 

The final strategy was to use Hugging Face's bert-base-uncased and pretrain it on 5000 words across 5 epochs. This result was also unsuccesful, yielding an average loss that plateaued around 0.2029. However, the predictions, using the softmax activation function, yieled only [PAD] tokens for BERT's .decode() function. Ultimately, no useful predictions were made by this model. 

Link to Hugging Face model upload for more details on training: https://huggingface.co/Brijhs/unmasker

In the future, providing a larger pretraining dataset for BERT or perhaps processing the definition and the masked word separately could improve accuracy. However, even running 5000 words across 5 epochs took just under 20 hours, so there were significant computational limits on training.
# Time Spent
On the final iteration of the model: ~15 hours of design

Previous iteration and attempts: ~10 hours

General learning of Pytorch, Hugging Face, and associated ML packages/concepts: ~10 hours
<<<<<<< HEAD

# Instructions for running
Model is built to be run from CLI using modelTester.py; it can be accompanied by a file name to test with, however by default it uses the provided testSet.csv as the test file to generate results. However, since this model does not work, the results are returned as empty lists. 
=======
>>>>>>> 5c4ef554301021adfa0b61d18243f38fe9c37261
