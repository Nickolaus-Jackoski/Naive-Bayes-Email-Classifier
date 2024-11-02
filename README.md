# Naive Bayes Email Classifier

## General Project Description
This repository hosts my academic project where I've developed a Naive Bayes classifier for email classification into two categories: spam and ham (non-spam). The project is focused on understanding and implementing the principles of probabilistic classification using Naive Bayes algorithms. The classifier is trained on a dataset of pre-classified emails, learning the distinction between spam and ham based on word frequencies.

## How to Run the Program

- **Folder Setup**: First ensure that the code and the .txt files are in the same folder.
- **Initializing Training and Testing**: Input the filenames for training and testing datasets for both spam and ham (not spam) emails.
- **Training the Classifier**: The classifier will process the training data to build its internal model.
- **Testing and Classification**: After training, the classifier will process the test data, outputting its predictions and the accuracy of its classifications.

## For SPAM:

- **P(spam)**: Probability of an email being spam. Calculated as the number of spam emails divided by the total number of emails (3/5).
- **P(viagra|spam)**: Probability of the word "viagra" appearing in a spam email. Smoothed to avoid zero probabilities (4/5).
- **P(phil|spam)**: Probability of "phil" in a spam email, smoothed (2/5).
- **P(~the|spam)**: Probability of "the" not appearing in a spam email, smoothed (1/5).
- The log probabilities are then summed: ln(3/5) + ln(4/5) + ln(2/5) + ln(1/5) = -3.260

## For HAM:

- **P(ham)**: Probability of an email being ham (2/5).
- **P(viagra|ham)**: Probability of "viagra" in a ham email, smoothed (1/4).
- **P(phil|ham)**: Probability of "phil" in a ham email, smoothed (3/4).
- **P(~the|ham)**: Probability of "the" not appearing in a ham email, smoothed (1/4).
- The log probabilities are summed: ln(2/5) + ln(1/4) + ln(3/4) + ln(1/4) = -3.977

## Comparing Scores and Classification

- The log probabilities for spam and ham are compared. The higher value indicates the more likely classification.
- In this example, -3.260 (SPAM) is higher than -3.977 (HAM), so the email is classified as spam.

## Why Log Probabilities?

Using log probabilities instead of direct probabilities is a common practice in machine learning for a few reasons:

1. **Avoiding Underflow**: Probabilities can become extremely small when multiplied together. Logarithms transform these small numbers into more manageable negative numbers.
2. **Addition Over Multiplication**: Adding log probabilities is computationally simpler and more stable than multiplying very small floating-point numbers.
3. **Interpretability**: Logarithms can make it easier to understand and compare the magnitudes of probabilities.

## Project Context
This project was a part of my coursework in the A.I. class I took at Rhodes College. It showcases my abilities in implementing machine learning algorithms and handling real-world data processing challenges, and demonstrating my understanding of the Naive Bayes algorithm in a practical, real-world application.
