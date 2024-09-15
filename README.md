# Natural Language Processing for Tweets Classification

This project focuses on building a robust model to classify tweets related to real-world disasters. The primary objective is to predict whether a tweet is about an actual disaster (label 1) or not (label 0).

## Overview

Leveraging modern natural language processing (NLP) techniques, this project implements a highly effective classification model using DeBERTa, distributed training with PyTorch, and advanced hyperparameter optimization strategies. 

## Key Technologies and Methodologies

1. **DeBERTa Model**
   - **Disentangled Attention Mechanism**: By separating content and positional information, this mechanism improves the model's understanding of the text's context.
   - **Enhanced Mask Decoder**: A more sophisticated approach to token reconstruction, especially useful for complex language tasks.

2. **PyTorch and Ray for Distributed Training**
   - PyTorch serves as the deep learning framework, providing flexibility.
   - **Ray** enables scalable, distributed training and hyperparameter tuning to optimize performance.

3. **Hyperparameter Optimization**
   - **ASHA Scheduler**: An efficient algorithm that prioritizes the most promising training configurations, saving time and resources.
   - **Optuna**: A hyperparameter tuning framework, applied to identify the most effective settings for model performance.

## Approach Summary

### 1. Data Preprocessing
The data preprocessing steps include:
- Cleaning tweet text by removing unwanted elements such as URLs and special characters.
- Tokenizing using DeBERTa’s tokenizer to maintain linguistic structure.
- Handling missing data effectively.

### 2. Custom Disaster Dataset Class
The `DisasterDataset` class was developed to handle text data preprocessing and conversion into a suitable format for the DeBERTa model. This class ensures the text is tokenized, padded, and truncated correctly, and prepares target labels as tensors for PyTorch.

### 3. Disaster Classification Model
The `DisasterModel` leverages DeBERTa as the backbone for extracting text features, followed by:
- **Dropout Layers** to prevent overfitting.
- **Fully Connected Layers** with ReLU activations to classify the tweet text.

### 4. Hyperparameter Tuning
- **ASHA Scheduler** dynamically allocates resources to the best-performing trials, improving efficiency.
- **Optuna** was employed to search for optimal hyperparameters like batch size, learning rate, and number of epochs, resulting in a well-tuned model.

### 5. Model Training and Evaluation
The model training process involved cross-validation and early stopping to avoid overfitting. Evaluation metrics like **accuracy**, **precision**, **recall**, and **F1-score** were used to assess performance on both training and validation datasets.

### 6. Prediction and Results
After training, the model was applied to test data, yielding strong performance in classifying disaster-related tweets.

## Results

This approach demonstrates the effectiveness of combining state-of-the-art NLP models with robust training and optimization techniques. The final model achieves high accuracy in classifying tweets based on whether they pertain to actual disasters.

## Conclusion

By integrating DeBERTa with PyTorch and leveraging distributed training through Ray, this project highlights an efficient solution to disaster-related tweet classification. The use of hyperparameter optimization techniques, including ASHA and Optuna, significantly improved model performance.

---

To run this project, you will need to follow these steps:

1. **Clone or download the repository** that contains the notebook.

2. **Install the required dependencies** by running:

    ```bash
    pip install -r requirements.txt
    ```

3. **Launch Jupyter Notebook**

4. **Open the notebook** (`deberta_tweets_classification.ipynb`) from the Jupyter interface.

5. **Run the cells** sequentially to execute the code and train the model.
