import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
def data_clean(data):
    data=data.drop_duplicates()
    return data
def univariate_analy(data):
    # Set the plot style
    sns.set_style("whitegrid")

# 1. Distribution of Age
    plt.figure(figsize=(8,5))
    sns.histplot(data['age'], bins=30, kde=True, color='blue')
    plt.title("Distribution of User Age", fontsize=14)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()
    # Count of users by gender
    sns.countplot(x='sex', data=data)
    plt.title('Gender Distribution')
    plt.show()

    # Count of users by browser
    sns.countplot(y='browser',data=data, order=data['browser'].value_counts().index)
    plt.title('Browser Usage Distribution')
    plt.show()

def bivariate_analy(data):
    # Heatmap of correlation
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Scatterplot: Age vs. Purchase Value
    sns.scatterplot(x='age', y='purchase_value', data=data)
    plt.title('Age vs. Purchase Value')
    plt.show()
    # Boxplot: Purchase Value by Gender
    sns.boxplot(x='sex', y='purchase_value', data=data)
    plt.title('Purchase Value by Gender')
    plt.show()

    # Violin Plot: Age vs. Class
    sns.violinplot(x='class', y='age', data=data)
    plt.title('Age vs. Fraud Class')
    plt.show()
    # Fraud class distribution by source
    sns.countplot(x='source', hue='class', data=data)
    plt.title('Fraud Class by Source')
    plt.show()
    # Crosstab: Browser vs. Fraud Class
    print(pd.crosstab(data['browser'], data['class']))
def encode_columns(data):
    cat_col=['browser','sex','age']
    #apply label encoder
    label_encoders={}
    for col in cat_col:
        le=LabelEncoder()
        data[col + '_encoded'] = le.fit_transform(data[col])
        label_encoders[col]=le
    return data