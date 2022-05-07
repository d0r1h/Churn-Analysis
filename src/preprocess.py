import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler 

import warnings
warnings.filterwarnings('ignore')

dataset = load_dataset("d0r1h/customer_churn")

def process():

    data = pd.DataFrame(dataset['train'])

    data['joined_through_referral'] = data['joined_through_referral'].replace('?',np.NaN)
    data['gender'] = data['gender'].replace('Unknown',np.NaN)
    data['referral_id'] = data['referral_id'].replace('xxxxxxxx',np.NaN)  
    data['medium_of_operation'] = data['medium_of_operation'].replace('?',np.NaN)  
    data['days_since_last_login'] = data['days_since_last_login'].replace(-999,np.NaN)  
    data['avg_time_spent']=data['avg_time_spent'].apply(lambda x:x if x>=0 else np.nan)
    data['points_in_wallet']=data['points_in_wallet'].apply(lambda x:x if x>=0 else np.nan)
    data['avg_frequency_login_days']=data['avg_frequency_login_days'].apply(lambda x:x if x!='Error' else -1)
    data['avg_frequency_login_days']=data['avg_frequency_login_days'].astype('float')
    data['avg_frequency_login_days']=data['avg_frequency_login_days'].apply(lambda x:x if x>=0 else np.nan)

    df_num = data.select_dtypes(include=np.number)
    df_cat = data.select_dtypes(include='object')

    Missing_cat = data[['gender','preferred_offer_types','region_category','joined_through_referral','medium_of_operation']]
    for i,col in enumerate(Missing_cat):
        data[col].fillna(data[col].mode()[0], inplace=True)
    Missing_num = data[['points_in_wallet','avg_time_spent','days_since_last_login','avg_frequency_login_days']]

    imputer = KNNImputer(n_neighbors=3)
    imputed_value=imputer.fit_transform(Missing_num)

    d1 = pd.DataFrame({
        'avg_frequency_login_days':imputed_value.T[0],
        'points_in_wallet':imputed_value.T[1],
        'days_since_last_login':imputed_value.T[2],
        'avg_time_spent':imputed_value.T[3]

    })

    data.drop(['avg_frequency_login_days','points_in_wallet','days_since_last_login','avg_time_spent'], axis=1, inplace=True)

    data = pd.concat([data, d1], axis=1)

    data['year']=data.joining_date.apply(lambda x:2021-int(x.split('-')[0]))
    data.drop(['security_no','joining_date','referral_id','last_visit_time'], axis=1, inplace=True)
    df_num=data.select_dtypes(include=[np.number]) 

    Q1 = data.quantile(0.25) 
    Q3 = data.quantile(0.75) 
    IQR = Q3 - Q1

    data_iqr = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)] 
    data_iqr.reset_index(inplace=True)
    data_iqr.drop('index',axis=1, inplace=True)

    df_cat = data_iqr[['gender','region_category','joined_through_referral','preferred_offer_types','medium_of_operation','internet_option','used_special_discount','offer_application_preference','past_complaint']]
    df_num = data_iqr.select_dtypes(include=np.number)

    orderencoding_membership_category = OrdinalEncoder(categories = [["No Membership", "Basic Membership", "Silver Membership", "Gold Membership","Platinum Membership","Premium Membership"]])
    orderencoding_complaint_status = OrdinalEncoder(categories = [["No Information Available", "Not Applicable", "Unsolved","Solved","Solved in Follow-up"]])
    labelencoder_feedback = LabelEncoder() 

    data_iqr['complaint_status'] = orderencoding_complaint_status.fit_transform(data_iqr['complaint_status'].values.reshape(-1,1)) 
    data_iqr['membership_category'] = orderencoding_membership_category.fit_transform(data_iqr['membership_category'].values.reshape(-1,1))
    data_iqr['feedback'] = labelencoder_feedback.fit_transform(data_iqr.feedback) 

    df_categorical = pd.get_dummies(df_cat, drop_first=True)

    df_final = pd.concat([df_categorical,df_num,data_iqr['membership_category'],data_iqr['complaint_status'],data_iqr['feedback']], axis=1)

    col = df_final[['age','days_since_last_login','avg_time_spent','avg_transaction_value','avg_frequency_login_days','points_in_wallet']]
    df_final.drop(['age','days_since_last_login','avg_time_spent','avg_transaction_value','avg_frequency_login_days','points_in_wallet'], axis=1, inplace=True)

    standard_scale = StandardScaler() 
    col1 = standard_scale.fit_transform(col) 
    df_scaled = pd.DataFrame(col1, columns=col.columns)

    data_final = pd.concat([df_final,df_scaled], axis=1)
    
    return data_final.to_csv("dataclean.csv", index=False, header=True)


if __name__ == "__main__":
    process()