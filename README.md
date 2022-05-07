# Customer Churn Prediction 

Churn prediction, or the task of identifying customers who are likely to discontinue use of a service, is an important and lucrative concern of any industry.

<img src = "https://github.com/d0r1h/Churn-Analysis/blob/main/static/churn_analysis.gif" width = 200>

### Description

This project is tasked to predict the churn score for a website based on features such as:

*    User demographic information
*    Browsing behavior
*    Historical purchase data among other information

This project aims to identify customers who are likely to leave so that we can retain them with certain incentives.


### DataSet:

* Dataset has been taken from a Hackathon, and raw dataset can be downloaded from here. [Link](https://www.hackerearth.com/problem/machine-learning/predict-the-churn-risk-rate-11-fb7a760d/)
* Cleaned and processed version of the data can be accessed from here. [Link](https://github.com/d0r1h/Churn-Analysis/blob/main/DataSet/churnclean.csv)
* Classes [Customer will EXIT(1) or NOT(0)] are properly balanced with 5:4 ratio


### Notebook:

Notebook contains the EDA, data processing, and model building ideas. 

| Notebook | Colab | Kaggle |
| ------ | ------ | ------ |
| Customer Churn Modeling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d0r1h/Churn-Analysis/blob/main/notebook/customer-churn.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/undersc0re/customer-churn) |
| Exploratory data analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d0r1h/Churn-Analysis/blob/main/notebook/eda-customer-churn.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/undersc0re/eda-customer-churn)


## Models

* The final model used is an ensemble of different classifiers such as:
    * KNN
    * Random Forest
    * AdaBoost
    * Xgboost


## Project Pipeline


<img src = "Project_Pipeline.png" width = 800>


### Techstack

**Python version** : 3.7 <br>
**Packages**: pandas, numpy, sklearn, xgboost, fastapi, seaborn <br>
**Cloud**: heroku


### Usage [running this locally]:

```python
conda create -n envname python=3.7
activate envname
git clone https://github.com/d0r1h/Churn-Analysis.git
cd Churn-Analysis
pip install -r requirements.txt
python app.py
```

To download dataset and preprocess automatically run following script

```python
!pip install datasets
!python src/preprocess.py
``` 

## Results 


* Even though Xgboost is giving good Test Accuracy of ~ 93% but we need to focus on the customers who are leaving i.e. class 1, so that we can retain them with some discount offer on membership.
* Ensemble methods (stack classifier) is having 94% of recall for predicting the customers who are likely to leave, higher than Xgboost.
* Following is confusion matrix of final classifier (stack ensemble) and xgboost classifier.

<img src = "https://github.com/d0r1h/Churn-Analysis/blob/main/static/stackclf.png" width = 300> <img src = "https://github.com/d0r1h/Churn-Analysis/blob/main/static/xgb.png" width = 300>

* Score table for different classifier

<figure>
<img src = "https://github.com/d0r1h/Churn-Analysis/blob/main/static/churn_score.png" width = 350>
<figcaption align = "center"></figcaption>
</figure>



## Inference Demo:

Application is deployed on heroku and can be accessed on https://churn01.herokuapp.com/ and sample data for the test app is [here](https://github.com/d0r1h/Churn-Analysis/blob/main/Examples/example_0.txt)


