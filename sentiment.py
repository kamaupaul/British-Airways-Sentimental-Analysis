# importing the libraries
import streamlit as st
import joblib
import string
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4') 
import contractions
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import matplotlib

matplotlib.use('Agg')

# preprocessing functions

# remove punctuations
exclude = string.punctuation
# removing punctuations from the reviews
def remove_punctuations(review):
    return review.translate(str.maketrans('','',exclude))

# tokenize the reviews
def tokenize_text(review):
    return word_tokenize(review)

# removing stop words
def remove_stopwords(tweet):
    stop_words = set(stopwords.words('english'))
    # Use list comprehension for efficient list creation
    new_tweet = [word for word in tweet if word not in stop_words]
    return " ".join(new_tweet)
    
# Lemmatizer.
def lem_words(tweet):
    word_lem = WordNetLemmatizer()
    return [word_lem.lemmatize(word) for word in tweet]
    

# removing the html tags
def remove_html(review):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', review)

# removing URL and @ sign

def preprocess_text_removingq_URLand_atsign(text):
    # Remove URLs
    clean_text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r'@[^\s]+', 'user', clean_text)
    # Other preprocessing steps like removing punctuation, converting to lowercase, etc.
    # ...
    return text



# loading the models

binary_model = joblib.load('logistic_regression_model.pkl')
multi_model = joblib.load('Multinomial_Naive_Bayes_Model.pkl')

# vectorizing the tweet for binary model
data = pd.read_csv('binary.csv')
binary_vectorizer = CountVectorizer()
binary_vectorizer.fit_transform(data['lemmatized_reviews_str'])

# vectorizing the tweet for multi model
data2 = pd.read_csv('multi.csv')
multi_vectorizer = CountVectorizer()
multi_vectorizer.fit_transform(data2['lemmatized_reviews_str'])

# we'll define 2 functions:
# 1. prediction generator.
# 2. the main function.

def prediction_generator_binary(text_vectorized):
    """function that takes the preprocessed text and passes to the model
    to see the sentiment prediction"""
    binary_predict = binary_model.predict(text_vectorized)

    if binary_predict[0] == 1:
        output_binary = 'Negative sentiment'
    else:
        output_binary = 'Positive sentiment'

    return output_binary

def prediction_generator_multi(text_vectorized):
    multi_predict = multi_model.predict(text_vectorized)
    
    # Extract the first element from the array
    prediction = multi_predict[0]

    if prediction == 1:
        output_multi = 'Negative sentiment'
    elif prediction == 3:
        output_multi = 'Positive sentiment'
    elif prediction == 2:
        output_multi = 'no emotion toward brand or product'
    else:
        output_multi = 'I cannot tell'
    
    return output_multi

# main function

def main():
    """handles the layout of the web app"""

    st.title('Testing binary and multi-class sentiment models')

    # Review input
    review = st.text_input('Enter Review or comments here')

    # Review preprocessing

    # lowecase
    review = review.lower()
    # removing html tags
    review = remove_html(review)
    # removing URL and @ sign
    review = preprocess_text_removingq_URLand_atsign(review)
    # expanding word
    #review = expand(review)
    # removing punctuations
    review = remove_punctuations(review)
    # tokenizing the data
    review = tokenize_text(review)
    # removing stopwords
    review = remove_stopwords(review)
    # re-tokenizing
    review = tokenize_text(review)
    # lemmatization
    review = lem_words(review)
    review = ' '.join(review)
    review_x = [review]  # Define review_x as a list with one element

   
    if st.button('Get result'):
	    # Ensure that review_x is a 2D array
	    review_x_vectorized_binary = binary_vectorizer.transform(review_x)  # Use the binary vectorizer
	    review_x_vectorized_multi = multi_vectorizer.transform(review_x)  # Use the multi vectorizer

	    test_result_binary = prediction_generator_binary(review_x_vectorized_binary)
	    test_result_multi = prediction_generator_multi(review_x_vectorized_multi)  # Use the correct vectorized data

	    st.success('Binary model says its a {0}'.format(test_result_binary))
	    st.success('Multi class model says its a {0}'.format(test_result_multi))

if __name__ == '__main__':
    main()

