import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"C:\Users\USER\Desktop\diabetes\diabetes.csv"
df = pd.read_csv(file_path)


# Замена нулевый значений
df["Glucose"] = df["Glucose"].replace(0, df["Glucose"].mean())
df["BloodPressure"] = df["BloodPressure"].replace(0, df["BloodPressure"].mean())
df["BMI"] = df["BMI"].replace(0, df["BMI"].mean())


# Матрица корреляций
corr_matrix = df.corr()

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title = "Матрица корелляций"
plt.show()

# Группировка по наличию диабета
diabetic = df[df["Outcome"] == 1]
non_diabetic = df[df["Outcome"] == 0]


# Гистограмма уровня глюкозы для пациентов с диабетом и без
sns.histplot(diabetic["Glucose"], color="red", label="Диабет", kde=True)
sns.histplot(non_diabetic["Glucose"], color="blue", label="Без диабета", kde=True)
plt.title = "Распределение уровня глюкозы"
plt.legend()
plt.show()


from scipy.stats import linregress


# Линейная регрессия: BMI
slope, intercept, r_value, p_value, std_err = linregress(df["BMI"], df["Outcome"])


# Линейная регрессия: Age
slope_age, intercept_age, r_value_age, p_value_age, std_err_age = linregress(
    df["Age"], df["Outcome"]
)

# Линейная регрессия: Glucose
slope_gluc, intercept_gluc, r_value_gluc, p_value_gluc, std_err_gluc = linregress(
    df["Glucose"], df["Outcome"]
)


# Строим модель прогноза
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df[["Glucose"]]
Y = df["Outcome"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, Y_train)

glucose_level = [[104]]  # Любое значение глюкозы
probability = model.predict_proba(glucose_level)[:, 1]


# Вывод результатов
print(f"r-value(BMI): {r_value:.2f}")
print(f"p-значение(BMI): {p_value:.2f}")

print(f"r-value(age): {r_value_age:.2f}")
print(f"p-значение(age): {p_value_age:.2f}")

print(f"r-value(glucose): {r_value_gluc:.2f}")
print(f"p-значение(glucose): {p_value_gluc:.2f}")

print(f"Вероятность диабета при уровне глюкозы {glucose_level}: {probability[0]:.2f}")
