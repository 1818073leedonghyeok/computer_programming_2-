import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

filename = "./data/dataset.csv"
column_names = ['MQ1', 'MQ2', 'MQ3', 'MQ4', 'MQ5', 'MQ6', 'CO2']
data = pd.read_csv(filename, names=column_names)

Y = data['CO2']
X = data.drop('CO2', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True, random_state=1)

model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(acc * 100))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=model.classes_, show=False)

plt.show()

plt.savefig("./result/bar_plot.png")
