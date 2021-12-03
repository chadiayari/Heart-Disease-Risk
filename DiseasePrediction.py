import pandas as pd
import matplotlib as mat
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

# Caracteristics values distribution in a diagram
heartData = pd.read_csv('heart.csv')
heartData.hist(figsize=(12,12))

X = heartData[[c for c in heartData.columns if c in ["age","cp","thalach"]]]
y = heartData[[c for c in heartData.columns if c == "target"]]
d_train, d_test, lab_train, lab_test = train_test_split(X, y, train_size=0.7, random_state=1)

abr = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=1)
abr.fit(d_train, lab_train)

# Caracteristics choice
fig = plt.figure(figsize=(30,14))
fn = ["age","cp","thalach"]
cn = ['0', '1']
tree.plot_tree(abr, feature_names = fn, class_names=cn, filled=True)
plt.show()

# Random Forests of 100 tree
FRcl = RandomForestClassifier(n_estimators=100)
FRcl.fit(d_train, lab_train.values.ravel())
# Apprentissage
y_pred_tr = FRcl.predict(d_train)
errTrain = 1-FRcl.score(d_train,lab_train)
# Test
errTest = 1-FRcl.score(d_test,lab_test)
print("Learning error rating: ", errTrain, " testing error rate: ", errTest)

scoreFRT =[]
scoreBg = []
for nbEstimateurs in range(10,1000,25):
  
# Random Forests
fr = RandomForestClassifier(n_estimators=nbEstimateurs,criterion="gini", max_depth=maxDep,min_samples_leaf=minSpLeaf,random_state=1)
fr.fit(d_train, lab_train)
# Test score
err = 1-fr.score(d_test, lab_test)
scoreFRT.append(err)

# Bagging Method
Bg = BaggingClassifier(tree.DecisionTreeClassifier(criterion="gini",max_depth=maxDep,min_samples_leaf=minSpLeaf), n_estimators=nbEstimateurs,random_state=1)
Bg.fit(d_train, lab_train)
scoreBg.append(1-Bg.score(d_test, lab_test))

# Showing curves
plt.plot(range(10,1000,25), scoreFRT,'o-', label='Random Forests')
plt.plot(range(10,1000,25), scoreBg,'x-', label='Bagging')
plt.xlabel("n_estimators")
plt.ylabel("Error Rating")
plt.legend()
plt.title('Error rating in function of number of estimators')
plt.grid(True)
plt.show()
