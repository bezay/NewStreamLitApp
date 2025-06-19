import pandas as pd
from sklearn.linear_model import LinearRegression
from skops.io import dump

# Load the dataset
df = pd.read_csv('housing_data.csv')

# Display the first few rows of the dataset
#print(df.head())
df.info()
#seperate the features and target variable
X = df[['Area_sqft', 'Bedrooms', 'Age_years']]
Y = df['Price_$']

#print(X, Y)

#Load and train the model
model = LinearRegression()
trained_model = model.fit(X, Y)

# Save the trained model to a file
dump(trained_model, 'housing_model.skops')
