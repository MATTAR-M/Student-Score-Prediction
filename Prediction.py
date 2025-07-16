import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Users/DELL/Downloads/archive (1)/StudentPerformanceFactors.csv")
df.head()
#checking the shape and dropping rows with missing values and columns with white space
df.shape
df=df.dropna()
df.columns = df.columns.str.strip()
sns.set(style="whitegrid")
df.isna()
#checking if there is a missing value
df.isna().sum()
#checking if there is a duplicated values
df.duplicated().sum()
#ignoring any warning in the output
import warnings
warnings.filterwarnings("ignore")
#Describtion of the data set
df.describe()
#relation between Hours Studied and Exam Scores
plt.figure(figsize=(9,5))
plt.scatter(df['Hours_Studied'],df['Exam_Score'])
plt.title('Studied hours vs Scores')
plt.xlabel('hours studied')
plt.ylabel('scores')
plt.grid(True)
plt.show()
#splitting the data into training data and target data
from sklearn.model_selection import train_test_split
x = df[['Hours_Studied']]
y = df['Exam_Score']
x_train ,x_test, y_train , y_test = train_test_split(x,y,test_size = 0.2, random_state=42)
#training linear regretion model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train,y_train)
slope = model.coef_[0]
intercept = model.intercept_
print(f"\n Equation: score = {slope: .2f} hours + {intercept: .2f}")
y_pred = model.predict(x_test)
#DataFrame that compare the Original data vs Predicted data
results = pd.DataFrame({
    'Hours Studied': x_test.values.flatten(),
    'Original Scores':y_test.values,
    'Predicted score':y_pred
})
#this is for better visualisation 
results = results.sort_values(by='Hours Studied').reset_index(drop= True)
display(results)
#evaluation using mse & R^2 Score
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
#relation between the Original data and the Predicatoin 
plt.figure(figsize=(9,5))
plt.scatter(x_test,y_test,label='The original scores')
plt.plot(x_test,y_pred,color='red',label='Predction line')  
plt.title('Studied hours vs Scores')
plt.xlabel('hours studied')
plt.ylabel('scores')
plt.grid(True)
plt.legend()
plt.show()