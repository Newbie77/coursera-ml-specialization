#import graphlab
#sales = graphlab.SFrame('home_data.gl/')
#train_data, test_data = sales.random_split(0.8, seed=0)
#sqft_model = graphlab.linear_regression.create(train_data, target="price", features=["sqft_living"])
#sqft_model.evaluate(test_data)
##import matplotlib.pyplot as plt
#print sqft_model.get("coefficients")

import graphlab
sales = graphlab.SFrame('home_data.gl/')


### Question 1
houses_with_highest_avg_house_sale_price = sales[sales["zipcode"] == "98039"]
print "Average price of houses having zipcode of 98039", houses_with_highest_avg_house_sale_price["price"].mean()
#2160606.5999999996


### Question 2
filtered_houses = sales[(sales['sqft_living'] > 2000) & (sales['sqft_living'] <= 4000)]
filtered_houses_fraction = filtered_houses.num_rows()*1.0/sales.num_rows()
print "Filtered houses fraction of all houses", filtered_houses_fraction
#0.42187572294452413


### Question 3
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
advanced_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode', 'condition', 'grade', 'waterfront', 'view', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
train_data, test_data = sales.random_split(0.8, seed=0)
my_features_model = graphlab.linear_regression.create(train_data, target="price", features=my_features)
advanced_features_model = graphlab.linear_regression.create(train_data, target="price", features=advanced_features, validation_set=None)
my_features_model_rmse = my_features_model.evaluate(test_data)["rmse"]
#178294.14621323228
advanced_features_model_rmse = advanced_features_model.evaluate(test_data)["rmse"]
#156831.1168021901
print "advanced_features_model_rmse lower by", advanced_features_model_rmse-my_features_model_rmse
