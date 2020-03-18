# Wine Variety Prediction using Natural Language Processing

This project utilizes Natural Language processing in the Big Data Environment to build a model that consumes user wine reviews 
from the ‘Kaggle Wine Dataset’ to predict a categorical variable, wine variety. Various supervised learning models were implemented 
and tuned for the same, namely, as Logistic Regression, Decision Tree Classifier and a bootstrap ensemble algorithm, Random Forest
Classifier. Additionally, the overall performance of the system was enhanced by creating an ensemble which sits on top of these 
trained models and selects the result from the model which has the highest confidence for every prediction. To discriminate
between the models weighted Precision, Recall, Weighted F-score, Accuracy, AUC-ROC and Confusion Matrix was used as the evaluation metrics.
