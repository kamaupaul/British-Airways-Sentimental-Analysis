# British-Airways-Sentimental-Analysis

![project_image](https://github.com/kamaupaul/British-Airways-Sentimental-Analysis/blob/main/data/Sentiment_Analysis_Projects.png)

## Table of Contents

- Sentiment Analysis Project
- Overview
- Problem Statement
- Data Understanding
- Methodology
- Evaluation
- Conclusion
- Recommendations
- Next Steps
- Installation
- Collaborators
- Repository Structure 

## Overview

British Airways is a major global airline that carries millions of passengers each year. The company collects a vast amount of feedback from its customers, both positive and negative. This feedback is valuable for understanding customer satisfaction and identifying areas for improvement. However, manually analyzing this feedback is a time-consuming and resource-intensive process.

Sentiment analysis is a technique that can be used to automatically analyze customer feedback and identify the overall sentiment of the feedback. This information can then be used to:

* Track changes in customer sentiment over time
* Identify areas where British Airways is excelling or falling short
* Develop targeted marketing campaigns
* Improve customer satisfaction
* 
## Problem Statement

The task is to build a sentiment analysis model capable of categorizing reviews sentiments about British Airways into positive, negative, or neutral categories. The model will be utilized to extract valuable insights for informed decision-making in business strategies and customer satisfaction.

## Data Understanding

The scope of this project will be limited to reviews of British Airways written in English. The reviews will be collected from online review sites, and travel forums.
If you navigate to this link: [https://www.airlinequality.com/airline-reviews/british-airways] you will see this data. Now, we can use Python and BeautifulSoup to collect all the links to the reviews and then to collect the text data on each of the individual review links.

## Methodology

### Data Preprocessing

I cleaned the raw text data by removing special characters, stopwords, and performed tokenization. Lemmatization was applied to reduce words to their common root form.

### Vectorization Techniques

- **Bag-of-Words (CountVectorizer):** Sklearn's CountVectorizer was used to convert the text data into a numerical representation, capturing the frequency of words in each document.

### Classification Models

I experimented with different classification algorithms, including:

1. **Logistic Regression Classifier:** Used for binary sentiment classification.
2. **Multinomial Naive Bayes and XGBoost:** Utilized for multiclass sentiment classification.

### Model Evaluation

The models were evaluated using accuracy as the primary metric to assess their ability to correctly predict sentiment.

## Evaluation

* Despite implementing resampling techniques to address class imbalance, the model's accuracy has not improved significantly. This suggests that other factors within the dataset may be limiting predictive performance.

* The binary logistic regression model performs best.
* For Multiclass Multinomial Naive Bayes Model was better.
  
1. **Logistic Regression Model:**
   - **Accuracy:** 0.84
   - **Precision, Recall, F1-Score:** Varies per class
   - **Comment:** The logistic regression model achieved a decent overall accuracy, but it's essential to examine class-specific metrics for a more detailed evaluation.

2. **Multinomial Naive Bayes Model:**
   - **Accuracy:** 0.78
   - **Precision, Recall, F1-Score:** Varies per class
   - **Comment:** The Multinomial Naive Bayes model showed lower accuracy compared to logistic regression. It's essential to investigate the class-specific metrics to understand its performance on individual classes.

3. **XGBoost Model:**
   - **Accuracy:** 0.72
   - **Precision, Recall, F1-Score:** Varies per class
   - **Comment:** XGBoost performance is slightly lower than logistic regression and Naive Bayes. Investigate class-specific metrics for more insights.

![image](https://github.com/kamaupaul/British-Airways-Sentimental-Analysis/assets/124625810/3bf46ce0-1ba3-4486-8608-31009592531b)

## Conclusion

1. **Customer Sentiment Overview:**
   - Provided a snapshot of the overall sentiment landscape, highlighting the prevailing positive sentiment but acknowledging a significant portion of negative sentiments.

2. **Commonly Mentioned Aspects:**
   - Recognized the importance of understanding the context behind frequently mentioned terms, emphasizing the need to distinguish emotional significance from neutral references.

## Recommendations
1. **In-depth Analysis of High-Frequency Terms:**
   - Suggested a detailed examination of high-frequency terms to unveil their emotional nuances, providing a more nuanced understanding of customer sentiment.

2. **Contextual Analysis:**
   - Recommended a context-specific analysis to distinguish between neutral and emotionally charged occurrences of common terms, ensuring a more accurate representation of sentiment.

3. **Stakeholder Engagement:**
   - Encouraged involvement of stakeholders from British Airways in the interpretation process to provide industry expertise and context to the sentiment analysis results.
## Next Steps

1. **Refinement of Sentiment Analysis:**
   - Plan to refine sentiment analysis algorithms to improve accuracy, incorporating feedback from stakeholders and validating results against manual assessments.

2. **Qualitative Analysis:**
   - Consideration of qualitative methods, such as sentiment-specific surveys or interviews, to gather deeper insights into customer emotions and preferences.

3. **Enhanced Topic Modeling:**
   - Expand topic modeling to uncover specific areas within highlighted themes that require attention, allowing for more targeted and actionable insights.

4. **Iterative Approach:**
   - Adopt an iterative approach to data collection and analysis, continuously refining methodologies based on ongoing feedback and emerging trends.

By addressing these findings, conclusions, limitations, recommendations, and next steps, the sentiment analysis project will evolve into a more robust and actionable tool for British Airways to enhance customer satisfaction and make informed business decisions.


### **Deployment**
I deployed the model using the framework STREAMLIT. The application takes a sentence and returns the sentiment. Link to web app 
[https://british-airways-sentimental-analysis-erqxrityuerb75kkgpwpmz.streamlit.app/](https://british-airways-sentimental-analysis-erqxrityuerb75kkgpwpmz.streamlit.app/)


## Installation

To install and run this project:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/your_username/your_repository.git
    ```

2. Navigate to the project's root directory:

    ```bash
    cd your-repository
    ```

3. Install the project dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the project:

    ```bash
    python src/sentiment_analysis.py
    ```

5. (Optional) Run the Streamlit app for interactive sentiment prediction:

    ```bash
    streamlit run src/streamlit_app.py
    ```

## Collaborators

- Paul Kamau

Feel free to contribute, report issues, or suggest improvements! Happy coding!

## Repository Structure

- `.gitignore`
- `CONTRIBUTING`
- `LICENSE.md`
- `src/`
  - `streamlit.py`
- `data/`
- `README.md`
- `requirements.txt`

Web app link: [Your Streamlit Web App Link]

## Project Status

The current version of the project is finished and ready for use. Improvements will be made later.
