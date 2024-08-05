import streamlit as st
import pandas as pd
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt
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
import sweetviz as sv
# st.set_option('deprecation.showPyplotGlobalUse', False)










# Page title and sidebar navigation
st.set_page_config(
    page_title="Machine Learning Algorithm Playground",
    page_icon="✅",
    layout="wide"
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

    .title {
        text-align: center;
        font-size: 34px; /* Adjust the font size as needed */
        font-weight: bold;
        text-transform: uppercase; /* Makes text uppercase */
        text-shadow: 2px 2px 3px rgba(0, 0, 0, 0.2); /* Adds a subtle 3D shadow effect */
        padding: 5px;
        display: inline-block;
        transition: color 0.3s, transform 0.3s;
        border: 1px solid #333;
        border-radius: 4px;
        
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        -moz-box-align: center;
    }
    .title:hover {
        color: #ff5733; /* Change the text color on hover */
        transform: scale(1.05); /* Scale up slightly on hover */
        text-shadow: 2px 2px 4px rgba(0.2, 0.1, 0, 0);
        box-shadow: 0px 4px 6px rgba(0.1, 0.2, 0, 0);
    }


    
    .subtitle {
        font-size: 24px;
        margin-bottom: 20px;
        text-align: justify;
        font-weight: 200;
        font-variant: normal;
        
    }
     .sub-header {
        font-size: 20px;
        font-weight: normal;
        margin-bottom: 10px;

    }
    .footer {
        font-size: 14px;
        color: #888;
        margin-top: 20px;
    }
    .stSelectbox {
        padding: 8px;
        background: linear-gradient(to bottom, #00a1e4 0%, #0077cc 100%);
        border: 1px solid #0077cc;
        border-radius: 5px;
        color:  #fff;
        box-shadow: 0 5px 15px rgba(0, 119, 204, 0.4);
        
    }

    .stSelectbox:hover {
        background: linear-gradient(to bottom, #0077cc 0%, #00a1e4 100%);
        box-shadow: 0 5px 15px rgba(0, 119, 204, 0.6);
    }

    .stSelectbox:focus {
        outline: none;
        background: linear-gradient(to bottom, #00a1e4 0%, #0077cc 100%);
        box-shadow: 0 5px 15px rgba(0, 119, 204, 0.8);
    }
    .stButton {
        display: inline-block;
        padding: 10px 20px;
        color: #000000a8;
        background-color: #ff5733;;
        border: #000000;
        position: relative;
        overflow: hidden;
        z-index: 1;
        text-align: center;
        cursor: pointer;
        }
    
    .stButton::before {
        content: '';
        background: #ff2e00;;
        border: 1px solid #d61d00;
        border-radius: 4px;
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        transform: scale(0, 1);
        transform-origin: 0% 50%;
        transition: transform 0.3s ease;
    }
    .stButton:hover::before {
        transform: scale(1, 1);
        transform-origin: 0% 50%;
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
    
    /* Checkbox styles */
    .stCheckbox label {
        position: relative;
        padding-left: 30px;
        cursor: pointer;
        user-select: none;
    }
    
    .stCheckbox label::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 20px;
        height: 20px;
        border: 1px solid #ccc;
        background-color: #fff;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        transition: all 0.3s;
    }
    
    .stCheckbox input[type="checkbox"] {
        display: none;
    }
    
    /* 3D effect when checked */
    .stCheckbox input[type="checkbox"]:checked + label::before {
        background-color: #0077cc; /* Change the background color when checked */
        border: 1px solid #005599; /* Change the border color when checked */
    }
    
    /* 3D effect on hover */
    .stCheckbox label:hover::before {
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.5);
    }
    .stNumberInput input[type="number"] {
    padding: 10px;
    border: 2px solid #0077cc;
    border-radius: 5px;
    box-shadow: 0 0 5px rgba(0, 0, 0, 0.2);
    }
    
    .stNumberInput input[type="number"]:hover {
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.4);
    }
    
    .stNumberInput input[type="number"]:focus {
        outline: none;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.6);
    }
    .stMultiSelect div[data-baseweb="select"] {
    border: 2px solid #0077cc;
    border-radius: 5px;
    box-shadow: 0 0 5px rgba(0, 0, 0, 0.2);
    }

    .stMultiSelect div[data-baseweb="select"]:hover {
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.4);
    }

    .stMultiSelect div[data-baseweb="select"]:focus {
        outline: none;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.6);
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
if "modified_test_data" not in st.session_state:
    st.session_state.modified_test_data = {}
    

st.markdown("<p class='welcome'>Welcome to the Machine Learning Algorithm Playground</p>",unsafe_allow_html=True)
# Add the header with links to LinkedIn and GitHub
st.markdown('<div class="sub-header">A Data Science Project by Akash Patil</div>', unsafe_allow_html=True)
st.markdown('''[LinkedIn](https://www.linkedin.com/in/akash-patil-985a7a179/)
[GitHub](https://github.com/akashpatil108)
''')

st.markdown("<div class='title'>Machine Learning Algorithm Playground</div>",unsafe_allow_html=True)


# Function to perform automated EDA using sweetviz
def perform_eda(data, target_feature):
    report = sv.analyze([data, "Original Data"], target_feature)
    return report

#This will create a horizontal rule
st.markdown("""<hr style="height:10px;border:none;color:#f84747d7;background-color:#f84747d7;" />""", unsafe_allow_html=True)

# Introduction section
st.markdown("This interactive web application is designed to help you explore and experiment with various machine learning algorithms for classification and regression tasks.")

st.subheader("Get Started")
st.markdown("Ready to get started? Follow these simple steps to explore the world of machine learning with our playground:")
st.markdown("1. **Upload Dataset:** Click on 'Upload Dataset' to upload your dataset in CSV format.")
st.markdown("2. **Explore Dataset:** Use the checkboxes to explore dataset characteristics like info, descriptive statistics, data distributions, correlation matrix, and pairplot.")
st.markdown("3. **Preprocess Data:** drop columns, and perform label encoding if needed.")
st.markdown("4. **Choose Algorithm:** Select a machine learning algorithm and customize hyperparameters.")
st.markdown("5. **Find Best Parameters:** Click 'Find Best Parameters' to perform hyperparameter tuning using GridSearchCV.")
st.markdown("6. **Evaluate Model:** Check accuracy metrics and classification report for classification or metrics like RMSE and R-squared for regression.")
st.markdown("7. **Test Model:** Upload a test dataset or manually input data for testing.")
st.markdown("8. **About this Project:** Learn more about the project and its purpose.")
st.markdown("9. **Get Help:** Reach out to the developer through LinkedIn or GitHub for assistance.")
st.markdown("10. **Enjoy Learning!:** Explore the world of data science without the hassle of complex coding.")
st.info("Note: You can upload your own data in a CSV file by selecting 'Upload your own dataset'.")




#This will create a horizontal rule
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" />""", unsafe_allow_html=True)
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
selected_dataset = st.selectbox("Select a Dataset", ["Select a Dataset", "Upload your own dataset 📂"] + dataset_names)

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
if selected_dataset == "Upload your own dataset 📂":
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
        
        st.write("Dataset:",original_data.head())
        
        st.subheader("Dataset Exploration Options:")
         # Display the dataset exploration options in a single row
        show_dataset_info, show_descriptive_statistics, show_data_distributions, show_correlation_matrix, show_pairplot = st.columns(5)
        
        with show_dataset_info:
            show_dataset_info=st.checkbox("Dataset info")
        with show_descriptive_statistics:
            show_descriptive_statistics=st.checkbox("Descriptive Statistics")
        with show_data_distributions:
            show_data_distributions= st.checkbox("Data Distributions") 
        with show_correlation_matrix:
            show_correlation_matrix= st.checkbox("Correlation Matrix")
        with show_pairplot:
            show_pairplot= st.checkbox("Pairplot (for smaller datasets)")


        # Display the selected exploration results
        
        # Create two columns
        SDO, SDS = st.columns(2)
        # Check if the user wants to specify delimiter
        with SDO:
            if show_dataset_info:
                    # Capture the output of DataFrame.info() in a buffer
                buffer = io.StringIO()
                original_data.info(buf=buffer)
    
                # Get the buffer content
                info_output = buffer.getvalue()
                st.subheader("Dataset info:")
                st.text(info_output)
        with SDS:
            if show_descriptive_statistics:
                st.subheader("Descriptive Statistics:")
                st.write(original_data.describe())

        with st.expander("Data Distributions", expanded=True):  # Set expanded to False by default
    
            if show_data_distributions:
                st.subheader("Data Distributions:")
                for column in original_data.select_dtypes(include=[np.number]).columns:
                    st.write(f"**{column}**")
                    # Create a figure with a specific size
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plt.hist(original_data[column], bins='auto', edgecolor='black', alpha=0.7)
                    st.pyplot()
                for column in original_data.select_dtypes(exclude='number').columns:
                    st.write(f"**{column}**")
                    # Create a figure with a specific size
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.countplot(x=column, data=original_data)
                    st.pyplot()

        with st.expander("Correlation matrix",expanded=True):

            if show_correlation_matrix:
                st.subheader("Correlation Matrix:")
                corr_matrix = original_data.corr()

                # Use seaborn heatmap to create a correlation matrix plot
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)

                # Display the plot using st.pyplot()
                st.pyplot(fig)


        with st.expander("pairplot",expanded=True):
            if show_pairplot and len(original_data) < 1000:  # Limit pairplot for smaller datasets
                st.subheader("Pairplot:")
                pairplot = sns.pairplot(original_data)
                st.pyplot(pairplot)

        if original_data is not None:
            st.write("Null Value Check:")
            null_values = original_data.isnull().sum()
            st.write(null_values)
        if null_values.any():
            st.warning("This dataset contains null values. Please consider handling null values before continuing with the analysis.")
            # Optionally, you can also display the columns with null values:
            st.write("Columns with Null Values:")
            st.write(null_values[null_values > 0].index.tolist())
        else:
            st.success("This dataset is free of null values. You can proceed with the analysis.")
        
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
                label_mappings = {}  # Dictionary to store label mappings

                for col in columns_to_encode:
                    modified_data[col] = label_encoder.fit_transform(modified_data[col])
                    # Create a mapping dictionary for this column
                    label_mappings[col] = {label: encoded_value for label, encoded_value in zip(original_data[col], modified_data[col])}

            # Store modified_data and label_mappings in session state
            st.session_state.modified_data = modified_data
            st.session_state.label_mappings = label_mappings

    
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
    st.write("Dataset:",original_data.head())
    st.subheader("Dataset Exploration Options:")
    # Display the dataset exploration options in a single row
    show_dataset_info, show_descriptive_statistics, show_data_distributions, show_correlation_matrix, show_pairplot = st.columns(5)

    with show_dataset_info:
        show_dataset_info=st.checkbox("Dataset info")
    with show_descriptive_statistics:
        show_descriptive_statistics=st.checkbox("Descriptive Statistics")
    with show_data_distributions:
        show_data_distributions= st.checkbox("Data Distributions") 
    with show_correlation_matrix:
        show_correlation_matrix= st.checkbox("Correlation Matrix")
    with show_pairplot:
        show_pairplot= st.checkbox("Pairplot (for smaller datasets)")

    # Create two columns
    SDO, SDS = st.columns(2)
    # Check if the user wants to specify delimiter
    with SDO:
        if show_dataset_info:
            # Capture the output of DataFrame.info() in a buffer
            buffer = io.StringIO()
            original_data.info(buf=buffer)

            # Get the buffer content
            info_output = buffer.getvalue()
            st.subheader("Dataset info:")
            st.text(info_output)
    with SDS:
        if show_descriptive_statistics:
            st.subheader("Descriptive Statistics:")
            st.write(original_data.describe())
    
    with st.expander("Data Distributions", expanded=True):  # Set expanded to False by default
    
        if show_data_distributions:
            st.subheader("Data Distributions:")
            for column in original_data.select_dtypes(include=[np.number]).columns:
                st.write(f"**{column}**")
                # Create a figure with a specific size
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.hist(original_data[column], bins='auto', edgecolor='black', alpha=0.7)
                st.pyplot()
            for column in original_data.select_dtypes(exclude='number').columns:
                st.write(f"**{column}**")
                # Create a figure with a specific size
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(x=column, data=original_data)
                st.pyplot()

    with st.expander("Correlation matrix",expanded=True):

        if show_correlation_matrix:
            st.subheader("Correlation Matrix:")
            corr_matrix = original_data.corr()

            # Use seaborn heatmap to create a correlation matrix plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)

            # Display the plot using st.pyplot()
            st.pyplot(fig)


    with st.expander("pairplot",expanded=True):
        if show_pairplot and len(original_data) < 1000:  # Limit pairplot for smaller datasets
            st.subheader("Pairplot:")
            pairplot = sns.pairplot(original_data)
            st.pyplot(pairplot)

    if original_data is not None:
        st.write("Null Value Check:")
        null_values = original_data.isnull().sum()
        st.write(null_values)

    if null_values.any():
        st.warning("This dataset contains null values. Please consider handling null values before continuing with the analysis.")
        # Optionally, you can also display the columns with null values:
        st.write("Columns with Null Values:")
        st.write(null_values[null_values > 0].index.tolist())
    else:
        st.success("This dataset is free of null values. You can proceed with the analysis.")


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
                label_mappings = {}  # Dictionary to store label mappings
        
                for col in columns_to_encode:
                    modified_data[col] = label_encoder.fit_transform(modified_data[col])
                    # Create a mapping dictionary for this column
                    label_mappings[col] = {label: encoded_value for label, encoded_value in zip(original_data[col], modified_data[col])}
        
            # Store modified_data and label_mappings in session state
            st.session_state.modified_data = modified_data
            st.session_state.label_mappings = label_mappings


   # Splitting data into features (X) and labels (y)
    X = st.session_state.modified_data.drop(columns=[target_feature]) if st.session_state.modified_data is not None else original_data.drop(columns=[target_feature])
    y = st.session_state.modified_data[target_feature] if st.session_state.modified_data is not None else original_data[target_feature]

    # Show the selected target feature (y)
    st.write(f"Selected Target Feature (y): {target_feature}")
    
    # Show the labeled features (X)
    st.write("Labeled Features (X):")
    st.write(X)

#This will create a horizontal rule
st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" />""", unsafe_allow_html=True)


st.write("### Automated Exploratory Data Analysis (EDA) 📊 Report:")
 # Display information about the Automated EDA
st.write("EDA is performed using the Sweetviz library, which generates a comprehensive report to analyze the dataset.")
# Display precautions and requirements
st.write("#### Precautions and Requirements:")
st.write("- Sweetviz expects the target feature to be either numerical or boolean.")
st.write("- Ensure that the selected target feature is suitable for analysis.")
# Display automated EDA report
if st.button("Generate Automated EDA Report"):
    if st.session_state.modified_data is not None:
        eda_report = perform_eda(st.session_state.modified_data, target_feature)
    elif original_data is not None:
        eda_report = perform_eda(original_data, target_feature)
    else:
        st.warning("No data available for EDA. Please upload a dataset or select a pre-loaded dataset.")

        # Display the EDA report
        st.write("Automated EDA Report:")
    st.write(eda_report.show_html())

#This will create a horizontal rule
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" />""", unsafe_allow_html=True)

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
#This will create a horizontal rule
st.markdown("""<hr style="height:8px;border:none;color:#333;background-color:#333;" />""", unsafe_allow_html=True)
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


#This will create a horizontal rule
st.markdown("""<hr style="height:8px;border:none;color:#333;background-color:#333;" />""", unsafe_allow_html=True)

st.header("Step 3: Train and Optimize Models")
st.markdown(
   """ "Select the parameters for training."
    "Then train the model." """)
hyperparameters = {}
if algorithm == "Random Forest":
        hyperparameters["n_estimators"] = st.number_input("Number of Estimators", min_value=1, value=100,key="a")
        hyperparameters["max_depth"] = st.number_input(" max_depth", min_value=1, value=100,key="b")
        hyperparameters["min_samples_split"] = st.number_input("min_samples_split", min_value=2, value=100,key="c")
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
        hyperparameters["min_samples_split"] = st.number_input("min_samples_split", min_value=2, value=100)
        hyperparameters["min_samples_leaf"] = st.number_input("min_samples_leaf", min_value=1, value=100)
        hyperparameters["random_state"] = st.number_input("Random State", min_value=0, value=42)
elif algorithm == "AdaBoost":
        hyperparameters["n_estimators"] = st.number_input("Number of Estimators", min_value=1, value=50,key="tp")
        hyperparameters["random_state"] = st.number_input("Random State", min_value=0, value=42,key="ta")
        hyperparameters["learning_rate"] = st.number_input("learning_rate", min_value=0.0, value=1.0,key="t")
elif algorithm == "Gaussian Naive Bayes":
        hyperparameters["var_smoothing"] = st.number_input("Variance Smoothing", min_value=1e-10, value=1e-9)
elif algorithm == "Decision Tree":
        hyperparameters["min_samples_split"] = st.number_input("Min Samples Split", min_value=2, value=2, key="ae")
        hyperparameters["min_samples_leaf"] = st.number_input("Min Samples Leaf", min_value=1, value=1, key="at")
        hyperparameters["criterion"] = st.selectbox("Criterion", ["gini", "entropy", "log_loss"])
        hyperparameters["random_state"] = st.number_input("Random State", min_value=0, value=42, key="ai")

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
        hyperparameters["criterion"] = st.selectbox("Criterion", ["absolute_error", "friedman_mse", "squared_error","poisson"])
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

#This will create a horizontal rule
st.markdown("""<hr style="height:8px;border:none;color:#333;background-color:#333;" />""", unsafe_allow_html=True)

# Check if the "Find Best Parameters" checkbox is selected
st.subheader(f"Step 4: Find Best Parameters and Cross validation score for {algorithm}")
if st.checkbox("Find Best Parameters", key="best"):
    # Select the algorithm
    if problem_statement == "Classification": 
        algorithm = st.selectbox("Select Algorithm", list(classification_algorithms.keys()), key="choose_algo")
        if algorithm == "Random Forest":
            custom_hyperparameters = {
                "n_estimators": st.multiselect("Random Forest - Number of Estimators", [100, 200, 300]),
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
                "learning_rate": st.multiselect("AdaBoost -learning_rate", [0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0,8.0]),
                "random_state": [42]
            }
        elif algorithm == "Gaussian Naive Bayes":
            # Define the range for var_smoothing
            var_smoothing_range = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

            # Allow the user to select multiple values
            selected_values = st.multiselect("Variance Smoothing", var_smoothing_range, default=[1e-9])

            # Use the selected values in your hyperparameters
            custom_hyperparameters = {
                "var_smoothing": selected_values
            }
        elif algorithm == "Decision Tree":
            custom_hyperparameters = {
                "max_depth": st.multiselect("Max Depth", [None, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]),
                "min_samples_split": st.multiselect("Min Samples Split", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "min_samples_leaf": st.multiselect("Min Samples Leaf", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
                "criterion": st.multiselect("Support Vector Machine - criterion", ["gini", "entropy", "log_loss"]),
                "random_state": [42]
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
            "n_estimators": st.multiselect("Random Forest - Number of Estimators", [100, 200, 300]),
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
        elif algorithm == "Ridge Regression":
            custom_hyperparameters = {
                "alpha": st.multiselect("Ridge Regression - Regularization Strength (alpha)", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
                "solver": st.multiselect("Ridge Regression - Solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]),
                "random_state": [42]
            }

        elif algorithm == "Lasso Regression":
                custom_hyperparameters = {
                "alpha": st.multiselect("Lasso Regression - Regularization Strength (alpha)", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
                "random_state": [42]
            }
        elif algorithm == "Support Vector Regression":
            custom_hyperparameters = {
            "C": st.multiselect("Support Vector Regression - Regularization Parameter (C)", [0.001, 0.01, 0.1, 1.0]),
            "kernel": st.multiselect("Support Vector Regression - Kernel", ["linear", "rbf", "poly", "sigmoid"]),
            "degree": st.multiselect("Support Vector Regression - Degree of Polynomial Kernel", [1,2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "gamma": st.multiselect("Support Vector Regression - Gamma", ["scale", "auto"] + [1e-3, 1e-4, 1e-5]),
            
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
            "learning_rate": st.multiselect("Gradient Boosting Regression - Learning Rate", [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            "max_depth": st.multiselect("Gradient Boosting Regression - Max Depth", [3, 4, 5, 6, 7, 8, 9, 10]),
            "random_state": [42]

            }
        elif algorithm == "AdaBoost Regression":
            custom_hyperparameters = {
            "n_estimators": st.multiselect("AdaBoost Regression - Number of Estimators", [50, 100, 200, 300, 400, 500]),
            "learning_rate": st.multiselect("AdaBoost Regression - Learning Rate", [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            "loss": st.multiselect("AdaBoost Regression - Loss Function", ["linear", "square", "exponential"]),
            "random_state": [42]
            }
        elif algorithm == "Decision Tree Regression":
            custom_hyperparameters = {
            "max_depth": st.multiselect("Decision Tree Regression - Max Depth", [None, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]),
            "min_samples_split": st.multiselect("Decision Tree Regression - Min Samples Split", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "min_samples_leaf": st.multiselect("Decision Tree Regression - Min Samples Leaf", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "criterion": st.multiselect("Decision Tree Regression - Criterion", ["absolute_error", "friedman_mse", "squared_error","poisson"]),
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

#This will create a horizontal rule
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" />""", unsafe_allow_html=True)

#: Test the Model
st.subheader("Step 5 : Test Model")
label_mappings = st.session_state.get("label_mappings", None)
if label_mappings is not None:
    st.header("Label Mappings")
    for col, mapping in label_mappings.items():
        st.subheader(f"Mapping for Column: {col}")
        st.write(mapping)

test_method = st.radio("Select Testing Method", ["Upload Test Dataset", "Manual Input"], key="test_method")
if test_method == "Upload Test Dataset":
    uploaded_test_file = st.file_uploader("Upload a CSV file for testing", type=["csv"])
    if uploaded_test_file is not None:
        # Read the uploaded test data into a DataFrame with optional delimiter and encoding
        Test_data = pd.read_csv(uploaded_test_file)
        test_data=Test_data.copy()
        # Display the uploaded test dataset
        st.write("Uploaded test dataset:")
        st.write(test_data)

        if target_feature in test_data.columns:
        # Drop the target feature from the dataset
            test_data = test_data.drop(columns=[target_feature])
            st.write("Target feature is droped")    
        # Option to apply column drop
        if st.checkbox("Apply Column Drop",key="cd"):
            columns_to_drop = st.multiselect("Select Columns to Drop", test_data.columns)
            test_data = test_data.drop(columns=columns_to_drop)
        # Option to apply label encoding
        if st.checkbox("Apply Label Encoding",key="le"):
            label_encoders = {}
            for col in test_data.columns:
                if test_data[col].dtype == 'object':
                    label_encoders[col] = LabelEncoder()
                    test_data[col] = label_encoders[col].fit_transform(test_data[col])
    
    
        #if apply_feature_scaling:
        scaling_method = st.selectbox("Select Feature Scaling Method", ["None", "StandardScaler", "Min-Max Scaling"])
        if scaling_method != "None":
            if scaling_method == "StandardScaler":
                scaler = StandardScaler()
            elif scaling_method == "Min-Max Scaling":
                scaler = MinMaxScaler()
            test_data = scaler.fit_transform(test_data)
        if st.button("Confirm Changes",key="utd"):
            #   You can store the modified test data for further use
            st.session_state.modified_test_data = test_data
    
            # Perform predictions using the trained classification or regression model
            if problem_statement == "Classification" and 'best_classifier' in st.session_state and 'best_hyperparameters' in st.session_state  and 'modified_test_data' in st.session_state:
                best_classifier = st.session_state.best_classifier
                classifier_params = st.session_state.best_hyperparameters
                test_data = st.session_state.modified_test_data
                best_classifier.set_params(**classifier_params)
                test_predictions = best_classifier.predict(test_data)
                st.subheader("Test Predictions")
                st.write("Predicted Labels:")
                st.write(test_predictions)
            elif problem_statement == "Regression" and 'best_classifier' in st.session_state and 'best_hyperparameters' in st.session_state  and 'modified_test_data' in st.session_state:
                best_regressor = st.session_state.best_regressor
                regressor_params = st.session_state.best_hyperparameters
                test_data = st.session_state.modified_test_data            
                best_regressor.set_params(**regressor_params)
                test_predictions = best_regressor.predict(test_data)
                st.subheader("Test Predictions")
                st.write("Predicted Values:")
                st.write(test_predictions)
    
            # Check if the target feature is available in the test dataset
            if target_feature in Test_data.columns:
                # Compare the target feature with the predicted values
                st.subheader("Comparison with Target Feature")
                target_values = Test_data[target_feature].values
                comparison_data = {
                    "Target Feature": target_values,
                    "Predicted Labels": test_predictions,
                }
                # Create a table for the comparison
                st.table(pd.DataFrame(comparison_data))

elif test_method == "Manual Input":
    if problem_statement == "Classification":

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
    elif problem_statement == "Regression":
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


#This will create a horizontal rule
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#f84747d7;" />""", unsafe_allow_html=True)


# About this Project Section
st.subheader("About this Project")

# Project Purpose
st.markdown("### Project Purpose:")
st.markdown("This project is designed as a friendly guide for beginners eager to explore the realms of data science and machine learning without delving into complex coding. Acting as a virtual assistant, it simplifies the step-by-step process, making data science accessible to all, whether on a computer or a mobile phone.")

# Q1
st.markdown("### Q1: What is this project all about?")
st.markdown("This project serves as a friendly tutor for those curious about data science and machine learning. It facilitates learning without the need for intricate coding, providing a seamless experience on both computers and phones.")

# Q2
st.markdown("### Q2: How can this project help me?")
st.markdown("This project is a valuable resource for beginners entering the data science and machine learning domain. It streamlines tasks like exploring data, preparing it for analysis, and experimenting with machine learning—no coding skills required.")

# Q3
st.markdown("### Q3: What can I do with this project?")
st.markdown("With this project, you can:")
st.markdown("- **Explore Data:** Easily visualize and understand your data.")
st.markdown("- **Preprocess Data:** Perform common preprocessing tasks like dropping columns and label encoding.")
st.markdown("- **Try Machine Learning:** Witness how machines can make predictions.")

# Q4
st.markdown("### Q4: Who is this project designed for?")
st.markdown("This project is ideal for:")
st.markdown("- **New Learners:** Perfect for those new to data and coding.")
st.markdown("- **Easy Access:** Usable on both computers and phones.")
st.markdown("- **No Coding Needed:** No coding expertise required.")

# Q5
st.markdown("### Q5: What are the limitations of this project?")
st.markdown("While this project is excellent for beginners and those eager to explore data science, it has some limitations:")
st.markdown("- **Simplicity:** It simplifies tasks, making it great for beginners but may not cover all advanced topics.")
st.markdown("- **Data Size:** Handling very large datasets can be slow and might not work well on mobile devices.")
st.markdown("- **Not for Experts:** If you're already an experienced data scientist, you might find it too basic for your needs.")
st.markdown("Remember, it's a fantastic starting point, but as you gain experience, you might explore more advanced tools in data science.")



st.markdown('''[LinkedIn](https://www.linkedin.com/in/akash-patil-985a7a179/)
[GitHub](https://github.com/akashpatil108)
''')

#This will create a horizontal rule
st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;" />""", unsafe_allow_html=True)



# Add a footer
st.markdown('<div class="footer">© 2023 Akash Patil </div> ', unsafe_allow_html=True)
