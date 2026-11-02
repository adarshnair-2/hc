# hc

5th-----
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Program5_StudentsPerformance.csv")

X = df[['reading score']]
y = df['math score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

new = pd.DataFrame({'reading score': [70]})
print("Prediction:", model.predict(new)[0])





6th---
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("Program6_diabetes.csv")

X = df[['Glucose']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

new = [[120]]
result = model.predict(new)

print("Prediction:", "Diabetic" if result[0] == 1 else "Non-diabetic")




7th------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Program7_Iris.csv")

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

new = [[5.0, 3.4, 1.5, 0.2]]
print("Prediction:", model.predict(new)[0])





8th--------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("Program8_Titanic.csv")

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Variance:", pca.explained_variance_ratio_)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("First Component")
plt.ylabel("Second Component")
plt.show()

new = [[3, 0, 25.0, 1, 0, 7.25, 0]]
new_scaled = scaler.transform(new)
new_pca = pca.transform(new_scaled)

result = model.predict(new_pca)
print("Prediction:", "Survived" if result[0] == 1 else "Did not survive")




9th------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("Program9_Mall_Customers.csv")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=5, random_state=42)
clusters = model.fit_predict(X_scaled)

print("Silhouette Score:", silhouette_score(X_scaled, clusters))

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.scatter(model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1], marker='x')
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()

new = [[50, 60]]
new_scaled = scaler.transform(new)
print("Cluster:", model.predict(new_scaled)[0])
