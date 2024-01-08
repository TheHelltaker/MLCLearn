from matplotlib import pyplot as plt 
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score 

df = pd.read_csv(r'C:\Users\divya\OneDrive\Desktop\MLC\LinearRegression\Salary\Salary Data.csv',chunksize=20)
print(df)

data = next(df)
test = next(df)

predictors =  ['YearsExperience']
outcome = 'Salary'

print("\nLinear regression")
SalMod = LinearRegression()
SalMod.fit(data[predictors],data[outcome])

for name , coef in zip(predictors , SalMod.coef_):
    print(f'{name}\'s coefficient : {coef}')


print("\nModel Assessment")
ModelPreds = SalMod.predict(test[predictors])
rmse = (mean_squared_error(test[outcome],ModelPreds))**(1/2)
R2 = r2_score(test[outcome], ModelPreds)

print(f'RMSE : {rmse}')
print(f'R2 score : {R2}')

print("\nmodel-plots")

plt.subplot(1,2,1)
plt.scatter(test[outcome], ModelPreds)
plt.plot(test[outcome], test[outcome], 'r')
plt.title('Actual vs Predicted Test Dataset')
plt.xlabel('Actual')
plt.ylabel('Predicted')


plt.subplot(1,2,2)
plt.scatter(data[predictors], data[outcome])
plt.plot(data[predictors] , data[predictors]*SalMod.coef_[0] + SalMod.intercept_ ,'r')
plt.title('Salaries')
plt.xlabel('Years of Experience')
plt.ylabel('Salaries')
plt.show()
