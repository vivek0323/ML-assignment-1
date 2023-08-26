import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt

file = r"C:\Users\akshi\Downloads\Lab Session1 Data.xlsx"
name = 'IRCTC Stock Price'
data = pd.read_excel(file, name)

price = data['Price']
mean = statistics.mean(price)
variance = statistics.variance(price)

print("Mean of Price data:", mean)
print("Variance of Price data:", variance)

data['Date'] = pd.to_datetime(data['Date'])
wednesday = data[data['Date'].dt.day_name() == 'Wednesday']
mean_wednesday = statistics.mean(wednesday['Price'])
print("Sample mean on Wednesdays:", mean_wednesday)
print("Population mean (overall mean):", mean)

april = data[data['Date'].dt.month == 4]
mean_april = statistics.mean(april['Price'])
print("Sample mean in April:", mean_april)

loss = len(data[data['Chg%'] < 0]) / len(data)
print("Probability of making a loss over the stock:", loss)

profit_wednesday = len(wednesday[wednesday['Chg%'] > 0]) / len(wednesday)
print("Probability of making a profit on Wednesday:", profit_wednesday)

conditional_probability = len(wednesday[wednesday['Chg%'] > 0]) / len(wednesday)
print("Conditional probability of making profit on Wednesday:", conditional_probability)

plt.scatter(data['Date'].dt.weekday, data['Chg%'])
plt.xlabel("Day of the Week")
plt.ylabel("Chg%")
plt.title("Chg% Data vs. Day of the Week")
plt.xticks(np.arange(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.show()
