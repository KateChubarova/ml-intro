import matplotlib
import pandas as pd

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

# Path of the file to read
flight_filepath = "train.csv"

# Read the file into a variable flight_data


train_data = pd.read_csv(flight_filepath, index_col="PassengerId")
# построить график выживых в зависимости от возраста
print(train_data.nunique())

men = len(train_data[train_data['Sex'] == 'male'])
women = len(train_data[train_data['Sex'] == 'female'])
men_p = men / len(train_data) * 100
women_p = women / len(train_data) * 100


saved = len(train_data[train_data['Survived'] == 1])
died = len(train_data[train_data['Survived'] == 0])
print(saved)
print(died)

print(f"Men on Titanic {men_p}%, women on Titanic {women_p}%")
