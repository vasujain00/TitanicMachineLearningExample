import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier


from sklearn.cross_validation import train_test_split


pd.options.mode.chained_assignment = None


from sklearn.externals import joblib

data = pd.read_csv("titanic_train.csv")
data.head()

data.columns

median_age = data['age'].median()
print("Median age is {}".format(median_age))

data['age'].fillna(median_age, inplace = True)
data['age'].head()

data_inputs = data[["pclass", "age", "sex"]]
data_inputs.head()

expected_output = data[["survived"]]
expected_output.head()

data_inputs["pclass"].replace("3rd", 3, inplace = True)
data_inputs["pclass"].replace("2nd", 2, inplace = True)
data_inputs["pclass"].replace("1st", 1, inplace = True)
data_inputs.head()

data_inputs["sex"] = np.where(data_inputs["sex"] == "female", 0, 1)
data_inputs.head()

inputs_train, inputs_test, expected_output_train, expected_output_test   = train_test_split (data_inputs, expected_output, test_size = 0.33, random_state = 42)

print(inputs_train.head())
print(expected_output_train.head())

rf = RandomForestClassifier (n_estimators=100)

rf.fit(inputs_train, expected_output_train)

accuracy = rf.score(inputs_test, expected_output_test)
print("Accuracy = {}%".format(accuracy * 100))

joblib.dump(rf, "titanic_model1", compress=9)

