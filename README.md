# h1n1-vaccine_Prediction

## Data Preprocessing:
- We started by loading the dataset and checking for any missing values or inconsistencies. We handled missing values by either imputing them or dropping rows with missing values. We converted the data types of relevant columns to the appropriate data types, such as converting categorical variables to the category data type.

## Data Visualization:
- We created various visualizations to explore the data and gain insights into the relationships between different features and the target variable (h1n1_vaccine). Some of the visualizations included bar charts, count plots, and histograms to understand the distributions and correlations between variables.

## Feature Engineering:
- We engineered new features from the existing ones, such as grouping age into brackets (age_bracket) and converting categorical variables into dummy variables using one-hot encoding. This transformation allowed us to use the categorical data in the model.

## Model Selection:
- Since the target variable (h1n1_vaccine) is binary (0 or 1), representing whether a person got vaccinated or not, we selected classification models for predicting vaccine uptake. We decided to use the following models:

  - Logistic Regression: A standard binary classification model that predicts the probability of an instance belonging to a particular class (vaccinated or not). It works well for linearly separable data and is easy to interpret.
  - Random Forest Classifier: An ensemble model that combines multiple decision trees to improve prediction accuracy and handle complex interactions between features. Random forests are robust against overfitting and can handle both numerical and categorical features.

  - AdaBoost Classifier: An ensemble learning method that boosts the performance of weak learners (e.g., Decision Trees) by giving more weight to difficult-to-predict samples. AdaBoost helps to improve generalization and reduce overfitting.

  - Gradient Boosting Classifier: Another ensemble learning method that builds multiple weak learners (e.g., Decision Trees) sequentially, with each learner focusing on the errors made by the previous learners. Gradient Boosting performs well on a wide range of datasets and often yields high accuracy.
  
  - Gaussian Naive Bayes Classifier: A probabilistic model based on Bayes' theorem, which assumes that features are conditionally independent given the class labels. It is simple, fast, and works well with high-dimensional data.
    
  - K Nearest Neighbors (KNN) Classifier: KNN is a non-parametric and lazy learning algorithm that classifies instances based on the majority class of its k-nearest neighbors in the feature space. It is effective for small to medium-sized datasets and can handle both numerical and categorical features. We used KNN to capture local patterns and relationships in the data.

  - Decision Tree Classifier: Similar to the Decision Tree Regressor, the Decision Tree Classifier is a non-linear classification algorithm that uses a tree-like structure to make predictions. It splits the data into subsets based on the features and classifies instances to the majority class in each leaf node. Decision trees are capable of capturing non-linear relationships in the data and are easy to interpret.

## Model Training and Evaluation:
- We split the dataset into training and testing sets to train the models and evaluate their performance on unseen data. After training the models, we used evaluation metrics such as accuracy, precision, recall, and F1-score to assess their performance.
 

## Hyperparameter Tuning:
- For ensemble models like Random Forest and Gradient Boosting, we performed hyperparameter tuning using techniques like GridSearchCV to find the best set of hyperparameters that optimize model performance.

## Final Model Selection:
- After comparing the performance of all the models, we selected the model with the highest accuracy and balanced precision-recall trade-off as the final model for predicting h1n1 vaccine uptake.

## Model Deployment:
- We saved the trained model for future use, such as making predictions on new data or deploying the model in a real-world application.
