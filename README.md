# Discovering Insights from British Airways Reviews Using AI and NLP   

## Overview
This project analyzes customer reviews from the British Airways website using Artificial Intelligence and Natural Language Processing (NLP) techniques. The objective was to extract meaningful insights, including sentiment and ratings, from unstructured text data and classify customer ratings using an Artificial Neural Network (ANN).

## Problem Statement
Customer feedback is vital for understanding service quality and customer satisfaction. British Airways reviews, available online, are often unstructured and lack explicit ratings or structured sentiments. This project aims to:
1. Extract reviews from the British Airways website.
2. Process the reviews to remove noise and structure the data.
3. Discover customer sentiments and ratings using advanced AI techniques.
4. Classify customer ratings using a machine learning approach.

---

## Features
- **Web Scraping**: Automated extraction of reviews from the British Airways website.
- **NLP Preprocessing**: Text cleaning, tokenization, and vectorization.
- **Sentiment Analysis**: Utilized TextBlob polarity to identify sentiments (positive, neutral, negative).
- **Ratings Discovery**: Applied K-means clustering to uncover customer ratings from textual data.
- **ANN Classification**: Developed an Artificial Neural Network to classify reviews into predicted ratings.

---

## Dataset
The dataset was scraped directly from the British Airways website. It contains one column:
- **Reviews**: Customer feedback in textual format.

---

## Technologies Used
### Programming Languages:
- Python

### Libraries and Tools:
- **Web Scraping**: `BeautifulSoup`, `requests`
- **NLP**: `nltk`, `TextBlob`, `scikit-learn`, `pandas`, `re`
- **Machine Learning**: `scikit-learn` (K-means clustering)
- **Deep Learning**: `TensorFlow`, `Keras`
- **Visualization**: `matplotlib`, `seaborn`

---

## Project Workflow
1. **Web Scraping**:
   - Extracted reviews from the British Airways website.
   - Stored the scraped data in a CSV file.

2. **NLP Preprocessing**:
   - Cleaned text data by removing HTML tags, emojis, and special characters.
   - Tokenized and lemmatized text for better processing.
   - Vectorized reviews using TF-IDF.

3. **Sentiment Discovery**:
   - Applied TextBlob to calculate polarity scores and categorized reviews into positive, neutral, or negative sentiments.

4. **Ratings Discovery**:
   - Used K-means clustering to group reviews into distinct clusters, each representing a rating (1 to 5).

5. **ANN Classification**:
   - Built a deep learning model to classify reviews into predicted ratings using extracted features.

6. **Evaluation and Visualization**:
   - Evaluated ANN model accuracy and visualized sentiment and ratings distribution.

---

## Results
- Sentiments (Positive, Neutral, Negative) discovered using TextBlob polarity scores.
- Ratings successfully classified using K-means clustering and ANN with high accuracy.
