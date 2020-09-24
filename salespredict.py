import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data= pd.read_csv("Sales.csv")

#Calculating Missing Values
print((data.isnull().sum()/len(data))*100)

#Replacing missing values of Item Weight
group_mean_weight = data.pivot_table(index = ["Item_Type"], values = "Item_Weight", aggfunc = [np.mean])
print(group_mean_weight)
mean_weight = group_mean_weight.iloc[:,[0][0]]
def missing_value(cols):
    item_type = cols[0]
    item_weight =cols[1]
    if pd.isnull(item_weight):
        return mean_weight[item_type] 
    return item_weight 

data["Item_Weight"] = data[["Item_Type","Item_Weight"]].apply(missing_value, axis = 1)

sns.countplot(data = data, x = "Outlet_Type",hue = "Outlet_Size")
plt.xticks(rotation =90)

def replace_size(cols):
    size = cols[0]
    ot_type = cols[1]
    if pd.isnull(size):
        if ot_type == "Supermarket Type1":
            return "Small"
        elif ot_type == "Supermarket Type2":
            return "Medium"
        elif ot_type == "Grocery Store":
            return "Small"
        elif ot_type == "Supermarket Type3":
            return "Medium"
    return size 

data["Outlet_Size"] = data[["Outlet_Size","Outlet_Type"]].apply(replace_size, axis = 1)

print(data["Item_Fat_Content"].unique())

data["Item_Fat_Content"] = data["Item_Fat_Content"].str.replace("LF", "low fat").str.replace("reg", "regular").str.lower()

print(data["Item_Fat_Content"].unique())

print("##Changing Item visibility 0 to mean")
mean_visibility = data.pivot_table(index = "Item_Identifier",  values = "Item_Visibility")

data.loc[(data["Item_Visibility"] == 0.0), "Item_Visibility"] = data.loc[(data["Item_Visibility"] == 0.0), "Item_Identifier"].apply(lambda x : mean_visibility.at[x, "Item_Visibility"])

cols = ['Item_Identifier', 'Item_Fat_Content',
       'Item_Type', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type']
#MAPPING EACH CATEGORICAL COLUMN WITH RESPECTIVE FREQUENCY OF THE VALUES IN THE COLUMNS
for i in cols:
    x  = data[i].value_counts().to_dict()
    data[i] = data[i].map(x)

print(data.head().transpose())

#Independent Variables:
x = data.drop("Item_Outlet_Sales", axis = 1) 

#Depenedent Variables 
y = data["Item_Outlet_Sales"].values.reshape(-1,1)

#Splitting The data  into Train and Test Dataset:
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x,y, test_size =0.20, random_state = 3)

#Applying Linear Regression Model
from sklearn.linear_model import LinearRegression
#regressor =DecisionTreeRegressor()
regressor =LinearRegression()
regressor.fit(x_train, y_train)

#Prediction
y_pred = regressor.predict(x_test)

#Accuracy of Model (Apply R2_score)
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred)*100)
