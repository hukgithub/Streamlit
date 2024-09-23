import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.write(''' 
# Exploring different ML Models and Datasets
and hosting it on Streamlit
''')

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ["Iris", "Breast Cancer", "Wine"]
)

classifier_name = st.sidebar.selectbox(
    "Select Classifier",
    ["SVM", "KNN", "Random Forest"]
)

def get_dataset(dataset_name):
    data = None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data 
    y = data.target
    return x, y

X, y = get_dataset(dataset_name)

st.write("Shape of Dataset: ", X.shape)
st.write("Number of Classes: ", len(np.unique(y)))

# Creating a Function Defining parameters for 3 classifiers
def add_parameter_ui(classifier_name):
    params = dict() # this will create an Empty Dictionary
    if classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C # its the degree of correct callsification
    elif classifier_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K # its the Number of Nearest Neighbours
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params['max_depth'] = max_depth # deoth of every Tree in Random Forest
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators # No. of Trees
    return params
# calling the function
params = add_parameter_ui(classifier_name)

# Creating a Function
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == "SVM":
        clf = SVC(C=params["C"], kernel='linear')
    elif classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth=params['max_depth'], random_state=1234)
    return clf

# To show the Code 
if st.checkbox('Show code'):
    with st.echo():
        # calling the fucntion
        clf = get_classifier(classifier_name, params)

        # splitting the Dataset for Training(80%) and Testing(20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

        # Training the Classifiers
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
# calling the fucntion
clf = get_classifier(classifier_name, params)

# splitting the Dataset for Training(80%) and Testing(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Training the Classifiers
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Checking Accuracy with a Scoring function
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

# Plotting Scatter Plot
# first we changed all features in 2 dimensional for plotting 
pca = PCA(2)
X_projected = pca.fit_transform(X)

# Slicing the Data in 0 and 1 dimension
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
            c = y, alpha = 0.8, 
            cmap = 'viridis')

plt.xlabel('Prinicipal Component 1')
plt.ylabel('Prinicipal Component 2')
plt.colorbar()

st.pyplot(fig)