# Scott Morgan
# MPG Calculator

import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

print("Reading data file, and performing preprocessing")
# opening file
file = open("data.csv")

# put the CSV into a dataframe we can work with
df = pd.read_csv(file)
# preprocessing stuff, scaling and label encoding
s = StandardScaler()
label_encoder = preprocessing.LabelEncoder()

maps = {}
# label encoding
# Also put all of the labels and their encodings into a single variable for access later
df["Make"] = label_encoder.fit_transform(df["Make"])
df["Make"].unique()
make_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
maps["Make"] = make_mapping

df["Model"] = label_encoder.fit_transform(df["Model"])
df["Model"].unique()
model_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
maps["Model"] = model_mapping

df["EngineFuelType"] = label_encoder.fit_transform(df["EngineFuelType"].astype(str))
df["EngineFuelType"].unique()
engine_fuel_type_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
maps["Engine Fuel Type"] = model_mapping

df["Transmission Type"] = label_encoder.fit_transform(df["Transmission Type"])
df["Transmission Type"].unique()
transmission_type_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
maps["Transmission Type"] = transmission_type_mapping

df["Driven_Wheels"] = label_encoder.fit_transform(df["Driven_Wheels"])
df["Driven_Wheels"].unique()
driven_wheels_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
maps["Driven Wheels"] = driven_wheels_mapping

df["Market Category"] = label_encoder.fit_transform(df["Market Category"].astype(str))
df["Market Category"].unique()
market_category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
maps["Market Category"] = market_category_mapping

df["Vehicle Size"] = label_encoder.fit_transform(df["Vehicle Size"])
df["Vehicle Size"].unique()
vehicle_size_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
maps["Vehicle Size"] = vehicle_size_mapping

df["Vehicle Style"] = label_encoder.fit_transform(df["Vehicle Style"])
df["Vehicle Style"].unique()
vehicle_style_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
maps["Vehicle Style"] = vehicle_style_mapping

# print(maps)

# remove the training and testing values so we can fit/compare results
length = len(df)

city_mpg = df['city mpg']
highway_mpg = df['highway MPG']
df = df.drop(columns=['city mpg'])
df = df.drop(columns=['highway MPG'])

# feature selection
# 0       make
# 1       model
# 2       year
# 3       engine fuel type
# 4       engine hp
# 5       engine cylinders
# 6       transmission type
# 7       driven wheels
# 8       num doors
# 9       market category
# 10      vehicle size
# 11      vehicle style
# 12      popularity
# 13      msrp

df = df.drop(df.columns[13], axis=1)
df = df.drop(df.columns[12], axis=1)
df = df.drop(df.columns[1], axis=1)
df = df.drop(df.columns[0], axis=1)

print("Scaling data, and storing City and Highway MPG values")

# Scaling
s.fit(df)
df = s.transform(df)

# 80 - 20 validation after the data is scaled
length = len(df)
x = (float)(length) * 0.8
x = int(x)

r = length - x
df = pd.DataFrame(df)
validation_df = pd.DataFrame()
train_df = df.loc[:x - 1]
validation_df = df.loc[x:]

# had some issues with NaN, so set them to zero
train_df = train_df.fillna(0)
validation_df = validation_df.fillna(0)

# city and highway training and validation data
train_city = city_mpg.loc[:x - 1]
validation_city = city_mpg.loc[x:]
train_highway = highway_mpg.loc[:x - 1]
validation_highway = highway_mpg.loc[x:]

# *********** DIFFERENT REGRESSION MODELS **************
# THE ONES THAT ARE COMMENTED OUT, RETURNED HIGHER MEAN ABSOLUTE ERROR VALUES (worse)

# regress_city = LogisticRegression(max_iter=10000)
# regress_highway = LogisticRegression(max_iter=10000)
# regress_city = Ridge(max_iter=10000)
# regress_highway = Ridge(max_iter=10000)
# regress_city = SGDRegressor(max_iter=10000)
# regress_highway = SGDRegressor(max_iter=10000)
# regress_city = LinearRegression()
# regress_highway = LinearRegression()
regress_city = GradientBoostingRegressor(max_depth=5, n_estimators=1000, min_samples_leaf=4)
regress_highway = GradientBoostingRegressor(max_depth=4, n_estimators=500)
# regress_city = KNeighborsRegressor()
# regress_highway = KNeighborsRegressor()
# regress_city = DecisionTreeRegressor()
# regress_highway = DecisionTreeRegressor()

print("Training the regression models, this may take a bit")
# fit the city/highway regression models with the correct data
regress_city.fit(train_df, train_city)
regress_highway.fit(train_df, train_highway)

print("Predicting values with the regression models")
# predict using all of our predicting things
city_score = regress_city.predict(validation_df)
highway_score = regress_highway.predict(validation_df)

# Print out the MAE for city/highway MPG
mae_city = mean_absolute_error(validation_city, city_score)
print("The Mean Absolute Error with Gradient Boosting Regressor values (CITY): \t", mae_city)
mae_highway = mean_absolute_error(validation_highway, highway_score)
print("The Mean Absolute Error with Gradient Boosting Regressor (HIGHWAY): \t\t", mae_highway)

print("Creating plots\nHighway MPG predictions vs Actual values stored in highway_results.png\n"
      "City MPG predictions vs Actual values stored in city_results.png")
# plot city data
fig, ax = plt.subplots()
y = validation_city
predicted = city_score
ax.scatter(y, predicted, color='#0000FF', alpha=0.05)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')

plt.savefig("city_results.png")

# plot highway data
fig, ax = plt.subplots()
y = validation_highway
predicted = highway_score
ax.scatter(y, predicted, color='#0000FF', alpha=0.05)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')

plt.savefig("highway_results.png")

print("Saving results to a CSV file: mpg_prediction_results.csv")
# put predictions into CSV file
mpg_results = pd.DataFrame()
mpg_results["Actual City MPG"] = validation_city
mpg_results["Predicted City MPG"] = city_score
mpg_results["Actual Highway MPG"] = validation_highway
mpg_results["Predicted Highway MPG"] = highway_score
mpg_results.to_csv("mpg_prediction_results.csv", index=False)


def print_dict(dict):
    for i, j in dict.items():
        print(j, " : ", i)
    return


go_again = 'y'
# car builder, run a loop so the user can choose to run the simulator again
while go_again == 'y' or go_again == 'Y':
    print()
    print("Welcome to build a car! This will allow you to select different features to try and increase your MPG")
    print("There will be sets of prompts that ask you to input a number to select a specific feature")
    print("At the end, the City and Highway MPG of a car with the specs you have given it will be displayed")
    print("**WARNING** there is no protection for entering a number outside of the range for specific options")
    print()

    year = input("Enter the year that the car was produced: ")

    print()
    print_dict(engine_fuel_type_mapping)
    fuel_type = input("Enter the number that corresponds to the type of fuel you want your car to use : ")

    print()
    horsepower = input("Enter the amount of HorsePower your car would have : ")

    print()
    num_cylinders = input("Enter the number of Cylinders that would be in your engine : ")

    print()
    print_dict(transmission_type_mapping)
    transmission_type = input("Enter the number that corresponds to the transmission type you want your car to have : ")

    print()
    print_dict(driven_wheels_mapping)
    driven_wheels = input("Enter the number that corresponds to the driven wheels you want your car to have : ")

    print()
    num_doors = input("Enter the number of doors you want your car to have (2 or 4) : ")

    print()
    print_dict(market_category_mapping)
    market_category = input("Enter the number that corresponds to the market category your car would be in : ")

    print()
    print_dict(vehicle_size_mapping)
    vehicle_size = input("Enter the number that corresponds to the vehicle size of your car : ")

    print()
    print_dict(vehicle_style_mapping)
    vehicle_style = input("Enter the number that corresponds to the vehicle style of your car : ")

    # store everything in a tuple
    user_ans = (year, fuel_type, horsepower, num_cylinders, transmission_type, driven_wheels, num_doors,
                market_category, vehicle_size, vehicle_style)
    # turn the tuple into a list, and then a numpy array
    convert = list()
    convert.append(user_ans)
    user_ans = np.asarray(convert)
    # scale the tuple
    user_ans = s.transform(user_ans)
    # predict based on our training data
    city_score = regress_city.predict(user_ans)
    highway_score = regress_highway.predict(user_ans)
    city = city_score[0]
    highway = highway_score[0]
    print()
    print("Awesome! Here are the predicted City and Highway MPG's based on the choices you made")
    print("City : ", city)
    print("Highway : ", highway)

    go_again = input("Would you like to re-run the Car Builder? (y/n) : ")

print("Thank you for using Car Builder!")
# close files
file.close()
