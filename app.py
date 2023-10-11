import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import (train_test_split,cross_val_score,KFold,LeaveOneOut)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from sklearn.preprocessing import LabelEncoder




# Page title and sidebar navigation
st.set_page_config(
    page_title="Machine Learning Algorithm Playground",
    page_icon="✅",
    layout="wide"
)
st.set_option('deprecation.showfileUploaderEncoding', False)

# Link to external CSS file (style.css)
st.markdown(
    """
    <link rel="stylesheet" href="style.css">
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
        
    }
    @media (max-width: 768px) {
        body {
            font-size: 14px;
            line-height: 20px;
        }
    }
     /* Define the animation */
    @keyframes slide {
        from {
            transform: translateY(0);
        }
        to {
            transform: translateY(-20px);
        }
    }
           
    
    .welcome{
    font-size: 25px;
    font-family: 'Times New Roman', Times, serif;
    font-style: italic;
    text-align: unset;
    animation: slide 1s alternate infinite;

    }

    /* Define the typewriter animation */
    @keyframes typewriter {
        from {
            width: 0;
        }
        to {
            width: 100%;
        }
    }
    .title {
        animation: typewriter 3s steps(60) 1s 1 normal both;
        white-space: nowrap;
        overflow: hidden;
        border-right: 3px solid #007acc;
        font-size: 46px;
        font-weight: bold;
        margin-bottom: 20px;
        text-align: center;
        touch-action: auto;
        color: rgb(25, 10, 190);
        border-color: #333;
        font-style: normal;
        border-radius: 2%;
        border-style: groove;
        background-color: #f59445;
        box-shadow: #524e4e;
        
    }
    .subtitle {
        font-size: 24px;
        margin-bottom: 20px;
        text-align: justify;
        font-weight: 200;
        font-variant: normal;
        
    }
     .sub-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .footer {
        font-size: 14px;
        color: #888;
        margin-top: 20px;
    }
    .stSelectbox {
        padding: 8px;
        border-radius: 2px;
        background-color: #9c9595;
        color: #dad3d3;
        
    }
    .stButton {
        display: flexbox;
        padding: 8px 16px;
        border-radius: 5px;
        color: #1b1717;
        border: #333;
        border-radius: 1%;
    }
    .stButton:hover {
        color: #007ACC;
        border-radius: 5%;
    }
    .stRadio {
        padding: 8px;
    }
    .stHeader {
        font-size: 36px;
        font-weight: bold;
        color: #007ACC;
    }
    .stSubheader {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    .stFileUploader {
        padding: 8px;
    }
    .stNumberInput {
        padding: 8px;
        border-radius: 5px;
        background-color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if 'Report' not in st.session_state:
    st.session_state.Report = []

if 'Regression_Report' not in st.session_state:
    st.session_state.Regression_Report = []

if 'selected_algorithm' not in st.session_state:
    st.session_state.selected_algorithm = []

if "X_train" not in st.session_state:
    st.session_state.X_train= []

if "X_test" not in st.session_state:
    st.session_state.X_test= []

if "y_train" not in st.session_state:
    st.session_state.y_train= []

if "y_test" not in st.session_state:
    st.session_state.y_test= []

if "X_train_scaled" not in st.session_state:
    st.session_state.X_train_scaled= []

if "X_test_scaled" not in st.session_state:
    st.session_state.X_test_scaled= []

if "custom_hyperparameters" not in st.session_state:
    st.session_state.custom_hyperparameters = {}

if "best_hyperparameters" not in st.session_state:
    st.session_state.best_hyperparameters = {}

if "best_classifier" not in st.session_state:
    st.session_state.best_classifier = {}

if "best_accuracy" not in st.session_state:
    st.session_state.best_accuracy = {}

if "best_reg_hyperparameters" not in st.session_state:
    st.session_state.best_reg_hyperparameters = {}

if "best_regressor" not in st.session_state:
    st.session_state.best_regressor = {}

if "best_rmse" not in st.session_state:
    st.session_state.best_rmse = {}

if "best_r2" not in st.session_state:
    st.session_state.best_r2 = {}

if "best_classification_report" not in st.session_state:
    st.session_state.best_classification_report = {}

st.markdown("<p class='welcome'>Welcome to the Machine Learning Algorithm Playground</p>",unsafe_allow_html=True)
# Add the header with links to LinkedIn and GitHub
st.markdown('<div class="sub-header">A Data Science Project by Akash Patil</div>', unsafe_allow_html=True)
st.markdown('''[LinkedIn](https://www.linkedin.com/in/akash-patil-985a7a179/)
[GitHub](https://github.com/akashpatil108)
''')

st.markdown("<div class='title'>Machine Learning Algorithm Playground</div>",unsafe_allow_html=True)

st.markdown("---")

# Introduction section
st.markdown("This interactive web application is designed to help you explore and experiment with various machine learning algorithms for classification and regression tasks.")

st.subheader("Get Started")
st.markdown("Ready to get started? Follow these simple steps to explore the world of machine learning with our playground:")
st.markdown("1. Upload your dataset in CSV format.")
st.markdown("2. Select the problem statement (classification or regression).")
st.markdown("3. Choose the machine learning algorithms you want to experiment with.")
st.markdown("4. Fine-tune hyperparameters and view cross-validation scores.")
st.markdown("5. Test your models and see how they perform on new data.")
st.markdown("6. Visualize the results and gain insights into your data.")

st.markdown("---")
st.header("Step 1: Load Data")
st.markdown(
    "Upload your preprocessed training dataset "
)
use_delimiter, use_encoding = st.columns(2)
# Additional options for delimiter and encoding
with use_delimiter:
    use_delimiter = st.checkbox("Specify Delimiter", key="use_delimiter")
with use_encoding:
    use_encoding = st.checkbox("Specify Encoding", key="use_encoding")
delimiter_options = [',', ';', '\t', '|']  # Common delimiter options
encoding_options = ['utf-8', 'ISO-8859-1', 'utf-16']  # Common encoding options
delimiter = None
encoding = None

# Create two columns
col1, col2 = st.columns(2)
# Check if the user wants to specify delimiter
with col1:
    if use_delimiter:
        delimiter = st.selectbox("Select Delimiter", delimiter_options)
# Check if the user wants to specify encoding
with col2:
    if use_encoding:
        encoding = st.selectbox("Select Encoding", encoding_options)
        
# Pre-loaded dataset options
dataset_names = sns.get_dataset_names()
dataset_options = {name: sns.load_dataset(name) for name in dataset_names}


# Add a selectbox to choose a dataset
selected_dataset = st.selectbox("Select a Dataset", ["Select a Dataset", "Upload your own dataset"] + dataset_names)

# Initialize variables for uploaded data
uploaded_file = None
selected_columns = []
columns_to_drop = []
columns_to_encode = []
# Initialize variables for uploaded data and modified data
uploaded_file = None
original_data = None
# Store modified_data in session state
if 'modified_data' not in st.session_state:
    st.session_state.modified_data = None
# Flags to track whether processes were applied
columns_dropped = False
label_encoding_applied = False

# Upload dataset
if selected_dataset == "Upload your own dataset":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    # Check if a file is uploaded
    if uploaded_file:
        # Read the uploaded data into a DataFrame with optional delimiter and encoding
        read_options = {}
        if delimiter:
            read_options['delimiter'] = delimiter
        if encoding:
            read_options['encoding'] = encoding
        
        original_data  = pd.read_csv(uploaded_file, **read_options)
        
        # Display the dataset
        st.write("Uploaded dataset:")
        st.write(original_data )

        target_feature = st.selectbox("Select Target Feature", original_data .columns, key="target_feature")

        # Option to drop columns
        if st.checkbox("Drop Columns"):
            columns_to_drop = st.multiselect("Select Columns to Drop", original_data.columns)
            columns_dropped = True
        
        # Option to apply label encoding
        if st.checkbox("Apply Label Encoding"):
            columns_to_encode = st.multiselect("Select Columns for Label Encoding", original_data.columns)
            label_encoding_applied = True
        
        if st.button("Confirm Changes"):
            # Create a copy of the original data for modifications
            modified_data = original_data.copy()
            
            # Drop selected columns if the user applied the process
            if columns_dropped:
                modified_data = modified_data.drop(columns=columns_to_drop)
            
            # Apply label encoding to selected columns if the user applied the process
            if label_encoding_applied:
                label_encoder = LabelEncoder()
                for col in columns_to_encode:
                    modified_data[col] = label_encoder.fit_transform(modified_data[col])
                    # Store modified_data in session state
            st.session_state.modified_data = modified_data
    
       # Splitting data into features (X) and labels (y)
        X = st.session_state.modified_data.drop(columns=[target_feature]) if st.session_state.modified_data is not None else original_data.drop(columns=[target_feature])
        y = st.session_state.modified_data[target_feature] if st.session_state.modified_data is not None else original_data[target_feature]

        # Show the selected target feature (y)
        st.write(f"Selected Target Feature (y): {target_feature}")
        
        # Show the labeled features (X)
        st.write("Labeled Features (X):")
        st.write(X)
        
elif selected_dataset != "Select a Dataset":
    # Load the selected pre-loaded dataset
    selected_dataset_name = selected_dataset
    original_data  = dataset_options[selected_dataset_name]

    # Display the selected pre-loaded dataset
    st.write(f"Selected Pre-loaded Dataset: {selected_dataset_name}")
    st.write("Preview of the selected pre-loaded dataset:")
    st.write(original_data )

    target_feature = st.selectbox("Select Target Feature", original_data.columns, key="target_feature")

    # Option to drop columns
    if st.checkbox("Drop Columns"):
        columns_to_drop = st.multiselect("Select Columns to Drop", original_data.columns)
        columns_dropped = True
    
    # Option to apply label encoding
    if st.checkbox("Apply Label Encoding"):
        columns_to_encode = st.multiselect("Select Columns for Label Encoding", original_data.columns)
        label_encoding_applied = True
    
    if st.button("Confirm Changes"):
        # Create a copy of the original data for modifications
        modified_data = original_data.copy()
        
        # Drop selected columns if the user applied the process
        if columns_dropped:
            modified_data = modified_data.drop(columns=columns_to_drop)
        
        # Apply label encoding to selected columns if the user applied the process
        if label_encoding_applied:
            label_encoder = LabelEncoder()
            for col in columns_to_encode:
                modified_data[col] = label_encoder.fit_transform(modified_data[col])
        # Store modified_data in session state
            st.session_state.modified_data = modified_data
    

   # Splitting data into features (X) and labels (y)
    X = st.session_state.modified_data.drop(columns=[target_feature]) if st.session_state.modified_data is not None else original_data.drop(columns=[target_feature])
    y = st.session_state.modified_data[target_feature] if st.session_state.modified_data is not None else original_data[target_feature]

    # Show the selected target feature (y)
    st.write(f"Selected Target Feature (y): {target_feature}")
    
    # Show the labeled features (X)
    st.write("Labeled Features (X):")
    st.write(X)

st.markdown("---")

st.header("Step 2: Select the Algorithm")
st.markdown(
    "Select the problem statement "
    "then select the algorithm."
)
# Problem Statement Selection
problem_statement = st.selectbox("Select Problem Statement", ["Classification", "Regression"],key="problem statement")
# Step 4: Algorithm Selection
if problem_statement == "Classification":
    st.subheader("Classification Algorithms")
    classification_algorithms = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }
    selected_algorithm_key = st.selectbox("Select Algorithm", list(classification_algorithms.keys()), key="select_algo")
    # Get the selected classifier class based on the key
    selected_algorithm = classification_algorithms[selected_algorithm_key]
elif problem_statement == "Regression":
    st.subheader("Regression Algorithms")
    regression_algorithms = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Support Vector Regression": SVR(),
        "Decision Tree Regression": DecisionTreeRegressor(),
        "Random Forest Regression": RandomForestRegressor(),
        "Gradient Boosting Regression": GradientBoostingRegressor(),
        "AdaBoost Regression": AdaBoostRegressor(),
        "K-Nearest Neighbors Regression": KNeighborsRegressor()
    }
    selected_algorithm_key = st.selectbox("Select Algorithm", list(regression_algorithms.keys()), key="select_algo")
    # Get the selected regressor class based on the key
    selected_algorithm = regression_algorithms[selected_algorithm_key]



# Save the selected classifier class to the session state
st.session_state.selected_algorithm = selected_algorithm
st.write(f"You have selected the {selected_algorithm_key} algorithm")
    # Display the saved algorithm value from session state
if 'selected_algorithm' in st.session_state:
    selected_algorithm = st.session_state.selected_algorithm
    st.write(f"Selected algorithm class: {selected_algorithm}")
#Train-Test Split
st.markdown("---")
st.subheader(" Train-Test Split")
st.write("Specify the split size (percentage) and the random state for reproducibility.")
split_size = st.slider("Select Split Size (%)", min_value=1, max_value=100, value=80)
random_state = st.number_input("Enter Random State", value=42)
# Perform train-test split
if X is not None and y is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split_size) / 100, random_state=random_state)
# Save the split data in session_state
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
# Display split sizes
if 'X_train' in st.session_state:
        X_train = st.session_state.X_train
        st.write(f" {X_train.shape}")
if 'X_test' in st.session_state:
        X_test = st.session_state.X_test
        st.write(f" {X_test.shape}")
hyperparameters = {}
st.write(selected_algorithm_key)
algorithm  = selected_algorithm_key
# Feature Scaling Selection
st.subheader("Feature Scaling")
st.write("Choose the feature scaling method:")
scaling_method = st.radio("Select Scaling Method", ["No Scaling", "StandardScaler", "Min-Max Scaling"], key="scaling_method")
# Conditional rendering based on the selected scaling method
if scaling_method == "StandardScaler":
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    st.write("Features are scaled using StandardScaler.")
    st.write("Feature Scaling is Successfully Done.")
elif scaling_method == "Min-Max Scaling":
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    st.write("Features are scaled using Min-Max Scaling.")
    st.write("Feature Scaling is Successfully Done.")
else:
    X_train_scaled = X_train
    X_test_scaled = X_test
   # Save the scaled data in session_state
    st.session_state.X_train_scaled = X_train_scaled
    st.session_state.X_test_scaled = X_test_scaled


st.markdown("---")
st.header("Step 3: Train and Optimize Models")
st.markdown(
    "Select the hyperparameters for training."
    "then train the model.")
hyperparameters = {}
if algorithm == "Random Forest":
        hyperparameters["n_estimators"] = st.number_input("Number of Estimators", min_value=1, value=100,key="a")
        hyperparameters["max_depth"] = st.number_input(" max_depth", min_value=1, value=100,key="b")
        hyperparameters["min_samples_split"] = st.number_input("min_samples_split", min_value=1, value=100,key="c")
        hyperparameters["min_samples_leaf"] = st.number_input("min_samples_leaf", min_value=1, value=100,key="d")
        hyperparameters["random_state"] = st.number_input("Random State", min_value=0, value=7)
elif algorithm == "Logistic Regression":
        hyperparameters["C"] = st.number_input("Regularization Parameter (C)", min_value=0.001, value=1.0,key="ab")
        hyperparameters["max_iter"] = st.number_input("Maximum Iterations", min_value=1, value=100,key="ac")
        hyperparameters["random_state"] = st.number_input("Random State", min_value=0, value=42)
elif algorithm == "Support Vector Machine":
        hyperparameters["C"] = st.number_input("Regularization Parameter (C)", min_value=0.001, value=1.0)
        hyperparameters["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly","sigmoid"])
        hyperparameters["random_state"] = st.number_input("Random State", min_value=0, value=42)
elif algorithm == "K-Nearest Neighbors":
        hyperparameters["n_neighbors"] = st.number_input("Number of Neighbors", min_value=1, value=5)
elif algorithm == "Gradient Boosting":
        hyperparameters["n_estimators"] = st.number_input("Number of Estimators", min_value=1, value=100)
        hyperparameters["min_samples_split"] = st.number_input("min_samples_split", min_value=1, value=100)
        hyperparameters["min_samples_leaf"] = st.number_input("min_samples_leaf", min_value=1, value=100)
        hyperparameters["random_state"] = st.number_input("Random State", min_value=0, value=42)
elif algorithm == "AdaBoost":
        hyperparameters["n_estimators"] = st.number_input("Number of Estimators", min_value=1, value=50,key="tp")
        hyperparameters["random_state"] = st.number_input("Random State", min_value=0, value=42,key="ta")
        hyperparameters["learning_rate"] = st.number_input("Random State", min_value=0.0, value=1.0,key="t")
elif algorithm == "Gaussian Naive Bayes":
        hyperparameters["var_smoothing"] = st.number_input("Variance Smoothing", min_value=1e-10, value=1e-9)
elif algorithm == "Decision Tree":
        hyperparameters["min_samples_split"] = st.number_input("Min Samples Split", min_value=2, value=2,key="ae"),
        hyperparameters["min_samples_leaf"] = st.number_input("Min Samples Leaf", min_value=1, value=1,key="at"),
        hyperparameters["criterion"] = st.selectbox("criterion", ["gini", "entropy","log_loss" ])
        hyperparameters["random_state"] =st.number_input("Random State", min_value=0, value=42,key="ai")
elif algorithm == "Ridge Regression":
        hyperparameters["alpha"] = st.number_input("Alpha (Regularization Strength)", min_value=0.001, value=1.0)
        hyperparameters["solver"] = st.selectbox("Solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
        hyperparameters["random_state"] = st.number_input("Random State", min_value=0, value=42)
elif algorithm == "Lasso Regression":
        hyperparameters["alpha"] = st.number_input("Alpha (Regularization Strength)", min_value=0.001, value=1.0)
        hyperparameters["random_state"] = st.number_input("Random State", min_value=0, value=42)
elif algorithm == "Support Vector Regression":
        hyperparameters["C"] = st.number_input("Regularization Parameter (C)", min_value=0.001, value=1.0)
        hyperparameters["kernel"] = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        hyperparameters["degree"] = st.number_input("Degree of Polynomial Kernel", min_value=1, value=3)
        hyperparameters["gamma"] = st.number_input("Gamma", min_value=0.0001, value=1.0)
elif algorithm == "K-Nearest Neighbors Regression":
        hyperparameters["n_neighbors"] = st.number_input("Number of Neighbors", min_value=1, value=5)
        hyperparameters["weights"] = st.selectbox("Weights", ["uniform", "distance"])
elif algorithm == "Decision Tree Regression":
        hyperparameters["max_depth"] = st.number_input("Max Depth", min_value=1, value=1)
        hyperparameters["min_samples_split"] = st.number_input("Min Samples Split", min_value=2, value=2)
        hyperparameters["min_samples_leaf"] = st.number_input("Min Samples Leaf", min_value=1, value=1)
        hyperparameters["criterion"] = st.selectbox("Criterion", ["mse", "friedman_mse", "mae"])
        hyperparameters["random_state"] = st.number_input("Random State", min_value=0, value=42)
elif algorithm == "Random Forest Regression":
        hyperparameters["n_estimators"] = st.number_input("Number of Estimators", min_value=1, value=100)
        hyperparameters["max_depth"] = st.number_input("Max Depth", min_value=1, value=10)
        hyperparameters["min_samples_split"] = st.number_input("Min Samples Split", min_value=2, value=2)
        hyperparameters["min_samples_leaf"] = st.number_input("Min Samples Leaf", min_value=1, value=1)
        hyperparameters["random_state"] = st.number_input("Random State", min_value=0, value=42)
elif algorithm == "Gradient Boosting Regression":
        hyperparameters["n_estimators"] = st.number_input("Number of Estimators", min_value=1, value=100)
        hyperparameters["learning_rate"] = st.number_input("Learning Rate", min_value=0.001, value=0.1)
        hyperparameters["max_depth"] = st.number_input("Max Depth", min_value=1, value=3)
        hyperparameters["min_samples_split"] = st.number_input("Min Samples Split", min_value=2, value=2)
        hyperparameters["min_samples_leaf"] = st.number_input("Min Samples Leaf", min_value=1, value=1)
        hyperparameters["random_state"] = st.number_input("Random State", min_value=0, value=42)
elif algorithm == "AdaBoost Regression":
        hyperparameters["n_estimators"] = st.number_input("Number of Estimators", min_value=1, value=50)
        hyperparameters["learning_rate"] = st.number_input("Learning Rate", min_value=0.0001, value=1.0)
        hyperparameters["loss"] = st.selectbox("Loss Function", ["linear", "square", "exponential"])
        hyperparameters["random_state"] = st.number_input("Random State", min_value=0, value=42)
Report = []
Regression_Report=[]
# Train a classification model
if st.button("Train Model"):
    if problem_statement == "Classification": 
        classifier = selected_algorithm 
        classifier.set_params(**hyperparameters)
        classifier.fit(X_train_scaled, y_train)
        y_train_pred = classifier.predict(X_test_scaled)  
        # Evaluate the model on the training data
        st.subheader(f"Model Performance on Training Data ({selected_algorithm})")
        Accuracy=accuracy_score(y_test, y_train_pred)
        st.write("Accuracy:",Accuracy)
        classification_rep = classification_report(y_test, y_train_pred)
        st.text("Classification Report:")
        st.text(classification_rep)
        Report.append(Accuracy)
        # Append the results to the session state Report
        st.session_state.Report.append({"Algorithm": selected_algorithm, "Accuracy": Accuracy, "Classification Report": classification_rep})
    elif problem_statement == "Regression": 
        classifier = selected_algorithm 
        classifier.set_params(**hyperparameters)
        classifier.fit(X_train_scaled, y_train)
        y_train_pred = classifier.predict(X_test_scaled)  
        mse = mean_squared_error(y_test, y_train_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_train_pred)
        st.write("Mean Squared Error (MSE):", mse)
        st.write("Root Mean Squared Error (RMSE):", rmse)
        st.write("R-squared (R2) Score:", r2)
        st.session_state.Regression_Report.append({"Algorithm": selected_algorithm, "MSE": mse, "RMSE": rmse, "R2 Score": r2})
# Display the saved Report
if problem_statement == "Classification":
    if st.checkbox("Show Report"):
        for result in st.session_state.Report:
            st.subheader(f"Model Performance on Training Data ({result['Algorithm']})")
            st.write("Accuracy:", result['Accuracy'])
            st.text("Classification Report:")
            st.text(result['Classification Report'])
elif problem_statement == "Regression":
    if st.checkbox("Show Report"):
        for result in st.session_state.Regression_Report:
            st.subheader(f"Model Performance on Training Data ({result['Algorithm']})")
            st.write("MSE:", result['MSE'])
            st.write("RMSE:", result['RMSE'])
            st.write("R2 Score:", result['R2 Score'])
custom_hyperparameters = ()

st.markdown("---")
# Check if the "Find Best Parameters" checkbox is selected
st.subheader(f"Step 4: Find Best Parameters and Cross validation score for {algorithm}")
if st.checkbox("Find Best Parameters", key="best"):
    # Select the algorithm
    if problem_statement == "Classification": 
        algorithm = st.selectbox("Select Algorithm", list(classification_algorithms.keys()), key="choose_algo")
        if algorithm == "Random Forest":
            custom_hyperparameters = {
                "n_estimators": st.multiselect("Random Forest - Number of Estimators", [1, 10, 100]),
                "max_depth": st.multiselect("Random Forest - Max Depth", [None, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]),
                "min_samples_split": st.multiselect("Random Forest - Min Samples Split", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "min_samples_leaf": st.multiselect("Random Forest - Min Samples Leaf", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
                "random_state": [7]
            }
        elif algorithm == "Logistic Regression":
            custom_hyperparameters = {
                "C": st.multiselect("Logistic Regression - Regularization Parameter (C)", [0.001, 0.01, 0.1, 1.0]),
                "max_iter": st.multiselect("Logistic Regression - Maximum Iterations", [100, 200, 300]),
                "random_state": [42]
            }
        elif algorithm == "Support Vector Machine":
            custom_hyperparameters = {
                "C": st.multiselect("Support Vector Machine - Regularization Parameter (C)", [0.001, 0.01, 0.1, 1.0]),
                "kernel": st.multiselect("Support Vector Machine - Kernel", ["linear", "rbf", "poly","sigmoid"]),
                "random_state": [42]
            }
        elif algorithm == "K-Nearest Neighbors":
            custom_hyperparameters = {
                "n_neighbors": st.multiselect("n_neighbors", [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,13,15,21,31]),
            }
        elif algorithm == "Gradient Boosting":
            custom_hyperparameters = {
                "n_estimators": st.multiselect("Gradient Boosting - Number of Estimators", [50, 100, 200, 300, 400, 500]),
                "min_samples_split": st.multiselect("Random Forest - Min Samples Split", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "min_samples_leaf": st.multiselect("Random Forest - Min Samples Leaf", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
                "random_state": [42]
            }
        elif algorithm == "AdaBoost":
            custom_hyperparameters = {
                "n_estimators": st.multiselect("AdaBoost - Number of Estimators", [50, 100, 200, 300, 400, 500]),
                "learning_rate": st.multiselect("AdaBoost -learning_rate", [0.0,0.2,0.4,0.6,0.8,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0,8.0]),
                "random_state": [42]
            }
        elif algorithm == "Gaussian Naive Bayes":
            custom_hyperparameters = {
                "var_smoothing": st.multiselect("Variance Smoothing", min_value=1e-10, max_value=1e-1, step=1e-10, value=1e-9)
            }
        elif algorithm == "Decision Tree":
            custom_hyperparameters = {
                "max_depth": st.multiselect("Max Depth", [None, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]),
                "min_samples_split": st.multiselect("Min Samples Split", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "min_samples_leaf": st.multiselect("Min Samples Leaf", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
                "criterion": st.multiselect("Support Vector Machine - criterion", ["gini", "entropy", "log_loss"]),
                "random_state": st.number_input("Random State", min_value=0, value=42)
            }
            # Check if the "Find Best Algorithm" button is clicked
        if st.button("Find Best Parameters"):
            # custom_hyperparameters = st.session_state.custom_hyperparameters
            classifier = selected_algorithm
            st.write(f"Finding best parameters for {classifier}...")
            # Perform hyperparameter tuning using GridSearchCV
            grid_search =GridSearchCV(classifier,custom_hyperparameters)
            grid_search.fit(X_train_scaled, y_train)
            # Get the best hyperparameters and classifier
            best_hyperparameters=grid_search.best_params_
            best_classifier=grid_search.best_estimator_
            st.write(grid_search.best_params_)
            # Save the best_classifier and best_hyperparameters in session_state
            st.session_state.best_hyperparameters = best_hyperparameters
            st.session_state.best_classifier = best_classifier
        if st.checkbox(f"find the best accuracy using best parameters "):
            if 'best_classifier' in st.session_state:
                best_classifier=st.session_state.best_classifier
            # Make predictions on the test data using the best classifier
            y_test_pred = best_classifier.predict(X_test_scaled)
            # Calculate accuracy using the best classifier
            accuracy = accuracy_score(y_test, y_test_pred)
            # Calculate the classification report using the best classifier
            classification_rep = classification_report(y_test, y_test_pred)
            # Save accuracy and classification report in session_state
            st.session_state.best_accuracy = accuracy
            st.session_state.best_classification_report = classification_rep
    elif problem_statement == "Regression": 
        algorithm = st.selectbox("Select Algorithm", list(regression_algorithms.keys()), key="choose_reg_algo")
        if algorithm == "Random Forest Regression":
            custom_hyperparameters = {
            "n_estimators": st.multiselect("Random Forest - Number of Estimators", [1, 10, 100]),
            "max_depth": st.multiselect("Random Forest - Max Depth", [None, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]),
            "min_samples_split": st.multiselect("Random Forest - Min Samples Split", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "min_samples_leaf": st.multiselect("Random Forest - Min Samples Leaf", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "random_state": [7]
            }
        elif algorithm == "Logistic Regression":
            custom_hyperparameters = {
            "C": st.multiselect("Logistic Regression - Regularization Parameter (C)", [0.001, 0.01, 0.1, 1.0]),
            "max_iter": st.multiselect("Logistic Regression - Maximum Iterations", [100, 200, 300]),
            "random_state": [42]
            }
        elif algorithm == "Support Vector Regression":
            custom_hyperparameters = {
            "C": st.multiselect("Support Vector Regression - Regularization Parameter (C)", [0.001, 0.01, 0.1, 1.0]),
            "kernel": st.multiselect("Support Vector Regression - Kernel", ["linear", "rbf", "poly", "sigmoid"]),
            "random_state": [42]
            }
        elif algorithm == "K-Nearest Neighbors Regression":
            custom_hyperparameters = {
            "n_neighbors": st.multiselect("K-Nearest Neighbors Regression - Number of Neighbors", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 21, 31]),
            "weights": st.multiselect("K-Nearest Neighbors Regression - Weights",["uniform", "distance"])
            }
        elif algorithm == "Gradient Boosting Regression":
            custom_hyperparameters = {
            "n_estimators": st.multiselect("Gradient Boosting Regression - Number of Estimators", [50, 100, 200, 300, 400, 500]),
            "min_samples_split": st.multiselect("Gradient Boosting Regression - Min Samples Split", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "min_samples_leaf": st.multiselect("Gradient Boosting Regression - Min Samples Leaf", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "random_state": [42]
            }
        elif algorithm == "AdaBoost Regression":
            custom_hyperparameters = {
            "n_estimators": st.multiselect("AdaBoost Regression - Number of Estimators", [50, 100, 200, 300, 400, 500]),
            "learning_rate": st.multiselect("AdaBoost Regression - Learning Rate", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            "random_state": [42]
            }
        elif algorithm == "Decision Tree Regression":
            custom_hyperparameters = {
            "max_depth": st.multiselect("Decision Tree Regression - Max Depth", [None, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]),
            "min_samples_split": st.multiselect("Decision Tree Regression - Min Samples Split", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "min_samples_leaf": st.multiselect("Decision Tree Regression - Min Samples Leaf", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "criterion": st.multiselect("Decision Tree Regression - Criterion", ["mse", "friedman_mse", "mae"]),
            "random_state": [42]
            }
            # Check if the "Find Best Algorithm" button is clicked
        if st.button("Find Best Parameters"):
            # Perform hyperparameter tuning using GridSearchCV
            regressor = selected_algorithm
            st.write(f"Finding best parameters for {regressor}...")
            grid_search = GridSearchCV(regressor, custom_hyperparameters)
            grid_search.fit(X_train_scaled, y_train)
            # Get the best hyperparameters and regressor
            best_hyperparameters = grid_search.best_params_
            best_regressor = grid_search.best_estimator_
            st.write("Best Hyperparameters:", best_hyperparameters)
            # Save the best_regressor and best_hyperparameters in session_state
            st.session_state.best_reg_hyperparameters = best_hyperparameters
            st.session_state.best_regressor = best_regressor
            # Check if the "find the best accuracy using best parameters" checkbox is selected
        if st.checkbox("Find the Best Accuracy Using Best Parameters"):
            if 'best_regressor' in st.session_state:
                best_regressor = st.session_state.best_regressor
                # Make predictions on the test data using the best regressor
                y_train_pred = best_regressor.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_train_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_train_pred)
                st.write("Mean Squared Error (MSE):", mse)
                st.write("Root Mean Squared Error (RMSE):", rmse)
                st.write("R-squared (R2) Score:", r2)
                # Save the best accuracy or metrics in session_state
                st.session_state.best_rmse=rmse
                st.session_state.best_r2 = r2
if st.button("display best result"):
    if problem_statement == "Classification":  
        # Display the best accuracy and best parameters
        if 'best_hyperparameters' in st.session_state and 'best_classifier' in st.session_state:
            st.write(f"Best Hyperparameters: {st.session_state.best_hyperparameters}")
            st.write(f"Best Accuracy: {st.session_state.best_accuracy}")
            st.text("Best Classification Report:")
            st.text(st.session_state.best_classification_report)
    elif problem_statement == "Regression": 
        if 'best_reg_hyperparameters' in st.session_state and 'best_regressor' in st.session_state:
            st.write(f"Best Hyperparameters: {st.session_state.best_hyperparameters}")
            st.write(f"best_rmse score: {st.session_state.best_rmse}")
            st.text("best_r2 score:")
            st.text(st.session_state.best_r2)
# Perform cross-validation using the best classifier and best parameters
st.markdown(f"Cross-Validation for {algorithm}")
if st.button("Perform Cross-Validation"):
    if problem_statement == "Classification":  
        best_classifier = st.session_state.best_classifier
        classifier_params = st.session_state.best_hyperparameters
        # Number of Cross-Validation Splits Selection
        st.subheader("Cross-Validation Settings")
        cv_splits = st.number_input("Number of Cross-Validation Splits", min_value=2, value=5)
        # Set the best parameters for the best classifier
        best_classifier.set_params(**classifier_params)
        # Calculate cross-validation scores (default scoring metric is accuracy)
        cv_scores = cross_val_score(
            best_classifier, X_train_scaled, y_train, cv=cv_splits, scoring='accuracy'
        )
        # Calculate the mean and standard deviation of cross-validation scores
        cv_mean_score = np.mean(cv_scores)
        cv_std_score = np.std(cv_scores)
        # Display cross-validation results
        st.subheader("Cross-Validation Results")
        st.write(f"Number of Cross-Validation Splits: {cv_splits}")
        st.write(f"Scoring Metric: Accuracy")  # Default scoring metric is accuracy
        st.write(f"Mean Accuracy: {cv_mean_score:.2f}")
        st.write(f"Standard Deviation: {cv_std_score:.2f}")
        st.write(f"Accuracy Scores: {cv_scores}")
    elif problem_statement == "Regression":
        best_regressor = st.session_state.best_regressor
        best_reg_hyperparameters = st.session_state.best_hyperparameters
        # Number of Cross-Validation Splits Selection
        st.subheader("Cross-Validation Settings")
        cv_splits = st.number_input("Number of Cross-Validation Splits", min_value=2, value=5)
        # Set the best parameters for the best regressor
        best_regressor.set_params(**best_reg_hyperparameters)
        # Calculate cross-validation scores (default scoring metric is R2)
        cv_scores = cross_val_score(
            best_regressor, X_train_scaled, y_train, cv=cv_splits, scoring='r2'
        )
        # Calculate the mean and standard deviation of cross-validation scores
        cv_mean_score = np.mean(cv_scores)
        cv_std_score = np.std(cv_scores)
        # Display cross-validation results
        st.subheader("Cross-Validation Results")
        st.write(f"Number of Cross-Validation Splits: {cv_splits}")
        st.write(f"Scoring Metric: R-squared (R2)")  # Scoring metric is R2
        st.write(f"Mean R2 Score: {cv_mean_score:.2f}")
        st.write(f"Standard Deviation: {cv_std_score:.2f}")
        st.write(f"R2 Scores: {cv_scores}")
#: Test the Model
st.markdown("---")
st.subheader("Step 5 : Test Model")
# Define problem statement
problem_statement = st.selectbox("Select Problem Statement", ["Classification", "Regression"])
if problem_statement == "Classification":
    # Classification Testing
    test_method = st.radio("Select Testing Method", ["Upload Test Dataset", "Manual Input"], key="test_method")
    if test_method == "Upload Test Dataset":
        uploaded_test_file = st.file_uploader("Upload a CSV file for testing", type=["csv"])
        if uploaded_test_file is not None:
            # Read the uploaded test data into a DataFrame with optional delimiter and encoding
            test_data = pd.read_csv(uploaded_test_file)
            # Display the uploaded test dataset
            st.write("Uploaded test dataset:")
            st.write(test_data)
            # Perform predictions using the trained classification model
            if 'best_classifier' in st.session_state and 'best_hyperparameters' in st.session_state:
                best_classifier = st.session_state.best_classifier
                classifier_params = st.session_state.best_hyperparameters
                best_classifier.set_params(**classifier_params)
                # Preprocess the uploaded test data (scaling if necessary)
                if scaling_method == "StandardScaler":
                    test_data_scaled = scaler.transform(test_data)
                elif scaling_method == "Min-Max Scaling":
                    test_data_scaled = scaler.transform(test_data)
                else:
                    test_data_scaled = test_data
                # Make predictions using the trained classification model
                test_predictions = best_classifier.predict(test_data_scaled)
                # Display the test predictions
                st.subheader("Test Predictions")
                st.write("Predicted Labels:")
                st.write(test_predictions)
    else:
        # Manual Input for Classification Testing
        st.subheader("Manual Input for Testing")
        st.write("Enter the data for testing manually:")
        # Create a DataFrame for manual input
        manual_input_df = pd.DataFrame(columns=X.columns)
        for feature in X.columns:
            manual_input = st.number_input(f"{feature}:", key=f"input_{feature}")
            manual_input_df[feature] = [manual_input]
        # Perform predictions using the trained classification model
        if 'best_classifier' in st.session_state and 'best_hyperparameters' in st.session_state:
            best_classifier = st.session_state.best_classifier
            classifier_params = st.session_state.best_hyperparameters
            best_classifier.set_params(**classifier_params)
            # Preprocess the manually input data (scaling if necessary)
            if scaling_method == "StandardScaler":
                manual_input_scaled = scaler.transform(manual_input_df)
            elif scaling_method == "Min-Max Scaling":
                manual_input_scaled = scaler.transform(manual_input_df)
            else:
                manual_input_scaled = manual_input_df
            # Make predictions using the trained classification model
            manual_test_predictions = best_classifier.predict(manual_input_scaled)
            # Display the manual test predictions
            st.subheader("Manual Test Predictions")
            st.write("Predicted Label:")
            st.write(manual_test_predictions)
else:
    # Regression Testing
    test_method = st.radio("Select Testing Method", ["Upload Test Dataset", "Manual Input"], key="test_method")
    if test_method == "Upload Test Dataset":
        uploaded_test_file = st.file_uploader("Upload a CSV file for testing", type=["csv"])
        if uploaded_test_file is not None:
            # Read the uploaded test data into a DataFrame with optional delimiter and encoding
            test_data = pd.read_csv(uploaded_test_file)
            # Display the uploaded test dataset
            st.write("Uploaded test dataset:")
            st.write(test_data)
            # Perform predictions using the trained regression model
            if 'best_regressor' in st.session_state and 'best_reg_hyperparameters' in st.session_state:
                best_regressor = st.session_state.best_regressor
                regressor_params = st.session_state.best_hyperparameters
                best_regressor.set_params(**regressor_params)
                # Preprocess the uploaded test data (scaling if necessary)
                if scaling_method == "StandardScaler":
                    test_data_scaled = scaler.transform(test_data)
                elif scaling_method == "Min-Max Scaling":
                    test_data_scaled = scaler.transform(test_data)
                else:
                    test_data_scaled = test_data
                # Make predictions using the trained regression model
                test_predictions = best_regressor.predict(test_data_scaled)
                # Display the test predictions
                st.subheader("Test Predictions")
                st.write("Predicted Values:")
                st.write(test_predictions)
    else:
        # Manual Input for Regression Testing
        st.subheader("Manual Input for Testing")
        st.write("Enter the data for testing manually:")
        # Create a DataFrame for manual input
        manual_input_df = pd.DataFrame(columns=X.columns)
        for feature in X.columns:
            manual_input = st.number_input(f"{feature}:", key=f"input_{feature}")
            manual_input_df[feature] = [manual_input]
        # Perform predictions using the trained regression model
        if 'best_regressor' in st.session_state and 'best_reg_hyperparameters' in st.session_state:
            best_regressor = st.session_state.best_regressor
            regressor_params = st.session_state.best_hyperparameters
            best_regressor.set_params(**regressor_params)
            # Preprocess the manually input data (scaling if necessary)
            if scaling_method == "StandardScaler":
                manual_input_scaled = scaler.transform(manual_input_df)
            elif scaling_method == "Min-Max Scaling":
                manual_input_scaled = scaler.transform(manual_input_df)
            else:
                manual_input_scaled = manual_input_df
            # Make predictions using the trained regression model
            manual_test_predictions = best_regressor.predict(manual_input_scaled)
            # Display the manual test predictions
            st.subheader("Manual Test Predictions")
            st.write("Predicted Value:")
            st.write(manual_test_predictions)

st.subheader("About this Project")
# Add the project purpose
st.markdown("Project Purpose:\n"
            "This project is like a friendly guide for beginners who want to learn how to use algorithms for data analysis. It helps them understand the step-by-step process of working with data without needing to write complex code. It's like having a virtual assistant that makes learning data science easy and accessible, even if you're using a mobile phone.")

# Add common questions and answers
st.markdown("Q1) What is this project all about?\n"
            "A1) This project is like a friendly tutor for people who are curious about data science and machine learning. It helps you learn without needing to write complex code. You can use it on your computer or even on your phone.")

st.markdown("Q2) How can this project help me?\n"
            "A2) This project is super useful if you're just starting to explore data science and machine learning. It makes it easy to look at data, make it ready for analysis (like removing stuff you don't need), and try out machine learning—no coding skills required.")

st.markdown("Q3) What can I do with this project?\n"
            "A3) With this project, you can:\n"
            "- Look at Data: You can easily see what data looks like.\n"
            "- Clean Data: You can get rid of data you don't want.\n"
            "- Try Machine Learning: You can see how machines can make predictions.")

st.markdown("Q4) Who is this project designed for?\n"
            "A4) This project is for:\n"
            "- New Learners: If you're new to data and coding, it's perfect for you.\n"
            "- Easy Access: You can use it on your computer or your phone.\n"
            "- No Coding Needed: You don't need to be a coding expert.\n"
            "- Learn at Your Pace: You can learn on your own schedule, no rush.")

st.markdown("Q5) What are the limitations of this project?\n"
            "A5) While this project is super helpful for new learners and those who want to explore data science easily, it has some limitations:\n"
            "- Simplicity: It simplifies complex data science tasks, which is great for beginners. However, it may not cover all advanced topics.\n"
            "- Dependency on Pre-built Models: It relies on pre-built machine learning models, limiting customization.\n"
            "- Data Size: Handling very large datasets can be slow and might not work well on mobile devices.\n"
            "- Not for Experts: If you're already an experienced data scientist, you might find it too basic for your needs.\n"
            "Remember, it's a fantastic starting point, but you might need to explore more advanced tools as you become more experienced in data science.")

# Add a footer
st.markdown('<div class="footer">© 2023 Akash Patil</div>', unsafe_allow_html=True)
