import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\pf3l1\Downloads\telecom.csv")
print(df)
print()
print(df.info())
print()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df.isnull().sum())
print()
df.dropna(inplace=True)
gender_mapping = {"Male": 0, "Female": 1}


phone_service_mapping = {"No": 0, "Yes": 1}
# Define a mapping dictionary for 'Contract' column
contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}

multiple_lines_mapping = {"No phone service": 0, "No": 1, "Yes": 2}

payment_method_mapping = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3,
}

churn_mapping = {"No": 0, "Yes": 1}
# Replace  values with their corresponding numeric values


df["PhoneService"] = df["PhoneService"].map(phone_service_mapping)


df["MultipleLines"] = df["MultipleLines"].map(multiple_lines_mapping)

df["Contract"] = df["Contract"].map(contract_mapping)

df["PaymentMethod"] = df["PaymentMethod"].map(payment_method_mapping)
df["Churn"] = df["Churn"].map(churn_mapping)


# Display the updated DataFrame
print(df.head())
print()

print(df.corr())


fig, axes = plt.subplots(1, 2, figsize=(14,8))

# Subplot 1: Scatter plot for MonthlyCharges vs. TotalCharges
sns.scatterplot(
    data=df,
    x="MonthlyCharges",
    y="TotalCharges",
    ax=axes[0],
    color="orange",
    alpha=0.7,
)
axes[0].set_xlabel("MonthlyCharges")
axes[0].set_ylabel("TotalCharges")
axes[0].set_title("MonthlyCharges vs. TotalCharges")

# Subplot 2: Scatter plot for tenure vs. TotalCharges
sns.scatterplot(
    data=df, x="tenure", y="TotalCharges", ax=axes[ 1], color="green", alpha=0.7
)
axes[1].set_xlabel("tenure")
axes[1].set_ylabel("TotalCharges")
axes[1].set_title("tenure vs. TotalCharges")


# Adjust layout and display the plots
print(plt.tight_layout())
print(plt.show())


from sklearn.linear_model import LinearRegression  # class has been created
from sklearn.model_selection import train_test_split

model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(
    df[["MonthlyCharges", "tenure"]],
    df["TotalCharges"],
    test_size=0.3,
    random_state=123,
)
print(x_train)
print()
print(x_test)
print()
print(y_train)
print()
print(y_test)
print()


model.fit(x_train, y_train)
pred = model.predict(x_test)
print(pred)
result = pd.DataFrame()
result["MonthlyCharges"] = x_test["MonthlyCharges"]
result["tenure"] = x_test["tenure"]
print(result)

result["Actual Sale"] = y_test
print(result)


result["Predicted Sale"] = pred
print(result)

result["Error"] = abs(result["Actual Sale"] - result["Predicted Sale"])
print(result)

result["Error Percent"] = (result["Error"] / result["Actual Sale"]) * 100
print(result)


mean_abs_error = result["Error"].mean()
print(mean_abs_error)

accuracy=100-mean_abs_error
print(accuracy)

from sklearn.metrics import r2_score
print(r2_score(y_test, pred))

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, pred))

root_mse = np.sqrt(mean_squared_error(y_test, pred))
print(root_mse)

print(model.score(x_test, y_test))

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, pred)
print(mae)
