# Fake News Detection
Fake news detection is the process of identifying false or misleading information published as legitimate news, typically online. With the rise of social media and fast-paced digital communication, fake news can spread rapidly and influence public opinion, politics, and societal behavior.To combat this, fake news detection uses techniques from natural language processing (NLP) and machine learning to automatically analyze and classify news content as real or fake. It involves examining the textual content, source credibility, writing style, and sometimes user behavior or metadata.This task is crucial in today’s digital world to ensure information integrity, combat misinformation, and promote trustworthy media.


# Process

1. Data Loading and Preprocessing:
- Loading the data from a CSV file is the initial step to get the raw information.
- Checking for missing values and handling them is crucial because many machine learning algorithms cannot handle missing data.
- Converting the 'label' column to numerical representation is necessary because machine learning models work with numerical inputs.
- Combining 'text' and 'title' into a single 'content' column creates a more comprehensive feature for the model to learn from, assuming both the title and the text of a news article contribute to determining if it's fake or real.
- Stemming is a text normalization technique that reduces words to their root form (e.g., "running," "runs," and "ran" become "run"). This helps reduce the vocabulary size and can improve the model's ability to generalize by treating words with similar meanings but different forms as the same.

2. Tokenizing and Padding:
- Tokenization breaks down the text into individual words or sub-word units (tokens). Each unique token is assigned a unique integer ID.
- Converting text to sequences of integers transforms the text data into a numerical format that can be fed into a neural network.
- Padding is necessary because neural networks, especially LSTMs, typically require input sequences of the same length. Padding adds zeros to the shorter sequences to match the length of the longest sequence

3. Splitting Data:
- Splitting the data into training and testing sets is a standard practice in machine learning to evaluate the model's performance on unseen data. The training set is used to train the model, and the test set is used to assess how well the trained model generalizes to new data. Using stratify=Y ensures that the proportion of real and fake news is the same in both training and testing sets, which is important for imbalanced datasets.

4. Building and Compiling the LSTM Model:
- Embedding Layer: This layer converts the integer sequences into dense vectors of fixed size. Words with similar meanings are expected to have similar vector representations. The input_dim is the size of the vocabulary (number of unique tokens), output_dim is the size of the embedding vectors, and input_length is the length of the input sequences.
- SpatialDropout1D: This layer performs dropout on the entire feature map, which helps to prevent overfitting by randomly setting a fraction of the input units to zero during training.
- LSTM Layer: Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that is well-suited for processing sequential data like text. LSTMs have internal mechanisms (gates) that allow them to capture long-term dependencies in the sequence, which is important for understanding the context of words in a sentence. The dropout and recurrent_dropout parameters further help in preventing overfitting.
- Dense Layer: This is a fully connected layer that takes the output of the LSTM layer and produces a single output.
- Sigmoid Activation: The sigmoid activation function is used in the output layer because this is a binary classification problem (real or fake). The sigmoid function outputs a value between 0 and 1, which can be interpreted as the probability of the input belonging to the positive class (real news in this case).
- Compilation: Compiling the model involves specifying the loss function, optimizer, and metrics.
               - Binary Crossentropy: This is the standard loss function for binary classification problems. It measures the difference between the predicted probabilities and the actual labels.
               - Adam Optimizer: Adam is an optimization algorithm that updates the model's weights during training to minimize the loss function.
               - Accuracy: Accuracy is a common metric used to evaluate classification models. It measures the proportion of correctly classified instances.

5. Training and Evaluation:
- Training the model involves feeding the training data to the model and adjusting the model's weights to minimize the loss function. The epochs parameter specifies the number of times the model will iterate over the entire training dataset, and batch_size determines the number of samples used in each training iteration. validation_data allows monitoring the model's performance on a separate dataset during training, which helps detect overfitting.
- Evaluating the model on the test data provides an unbiased estimate of how well the model will perform on unseen data.

6. Predictions and Accuracy Calculation:
- Making predictions on the training and test data involves using the trained model to predict the labels for these datasets.
- Calculating the accuracy score on both the training and test sets provides a measure of the model's performance on both seen and unseen data, allowing for the identification of potential overfitting.

This entire process aims to build a model that can accurately classify news articles as real or fake based on their textual content.


# Conclusion

Based on the output of the code:
- The model achieved a high accuracy of approximately 99.27% on the training data.
- However, the accuracy on the test data is significantly lower, around 87.77%.

This suggests that the model might be overfitting to the training data. Overfitting occurs when a model learns the training data too well, including the noise and specific patterns, which negatively impacts its ability to generalize to unseen data (the test set).

To improve the model's performance on unseen data, you could consider techniques to mitigate overfitting, such as:
- Increasing the dropout rates in the LSTM and SpatialDropout1D layers.
- Adding more data if possible.
- Using regularisation techniques.
- Using a simpler model architecture.
- Early stopping during training.
