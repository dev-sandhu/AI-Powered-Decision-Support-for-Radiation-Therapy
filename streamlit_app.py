import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load synthetic data
def load_data():
    np.random.seed(42)
    n_samples = 500

    # Generate synthetic data
    age = np.random.randint(50, 80, n_samples)
    prostate_volume = np.random.uniform(20, 50, n_samples)
    psa_levels = np.random.uniform(1, 20, n_samples)
    gleason_score = np.random.randint(6, 10, n_samples)
    tumor_grade = np.random.choice(['Low', 'Intermediate', 'High'], n_samples)
    tumor_location = np.random.choice(['Inside Prostate', 'Outside Prostate'], n_samples)
    prostate_boundary_distance = np.random.uniform(0.5, 3.0, n_samples)
    erectile_vessel_distance = np.random.uniform(0.5, 2.5, n_samples)
    treatment_type = np.random.choice(['Brachytherapy', 'External Beam Radiation', 'Both'], n_samples)
    radiation_dose_prostate = np.random.uniform(60, 120, n_samples)
    radiation_dose_vessels = np.random.uniform(0, 20, n_samples)
    vessel_sparing_flag = np.random.choice([0, 1], n_samples)
    erectile_function_post = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])

    data = pd.DataFrame({
        'Age': age,
        'Prostate Volume (cc)': prostate_volume,
        'PSA Levels': psa_levels,
        'Gleason Score': gleason_score,
        'Tumor Grade': tumor_grade,
        'Tumor Location': tumor_location,
        'Prostate Boundary Distance (cm)': prostate_boundary_distance,
        'Erectile Vessel Distance (cm)': erectile_vessel_distance,
        'Treatment Type': treatment_type,
        'Radiation Dose Prostate (Gy)': radiation_dose_prostate,
        'Radiation Dose Vessels (Gy)': radiation_dose_vessels,
        'Vessel Sparing': vessel_sparing_flag,
        'Erectile Function Post-Treatment': erectile_function_post
    })
    
    return data

# Preprocess the data
def preprocess_data(data):
    label_encoders = {}
    for column in ['Tumor Grade', 'Tumor Location', 'Treatment Type']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    scaler = StandardScaler()
    numeric_columns = ['Age', 'Prostate Volume (cc)', 'PSA Levels', 'Gleason Score',
                       'Prostate Boundary Distance (cm)', 'Erectile Vessel Distance (cm)',
                       'Radiation Dose Prostate (Gy)', 'Radiation Dose Vessels (Gy)']
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data

# Train models
def train_model(data):
    X = data.drop(columns=['Erectile Function Post-Treatment'])
    y = data['Erectile Function Post-Treatment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    
    # Train random forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return log_reg, rf_model, X_test, y_test

# Streamlit App
def main():
    st.title("AI-Powered Decision Support for Radiation Therapy")

    # Load and preprocess data
    data = load_data()
    data = preprocess_data(data)

    # Train the models
    log_reg, rf_model, X_test, y_test = train_model(data)

    # Allow user to input custom data for predictions
    st.sidebar.header("Input Patient Data")
    age = st.sidebar.slider('Age', 50, 80, 60)
    psa_levels = st.sidebar.slider('PSA Levels', 1.0, 20.0, 5.0)
    prostate_volume = st.sidebar.slider('Prostate Volume (cc)', 20.0, 50.0, 30.0)
    gleason_score = st.sidebar.slider('Gleason Score', 6, 10, 7)
    radiation_dose_prostate = st.sidebar.slider('Radiation Dose Prostate (Gy)', 60.0, 120.0, 90.0)
    radiation_dose_vessels = st.sidebar.slider('Radiation Dose Vessels (Gy)', 0.0, 20.0, 10.0)

    # Prediction button
    if st.sidebar.button("Predict Outcome"):
        # Create dataframe with the custom input
        input_data = pd.DataFrame({
            'Age': [age],
            'Prostate Volume (cc)': [prostate_volume],
            'PSA Levels': [psa_levels],
            'Gleason Score': [gleason_score],
            'Tumor Grade': [1],  # Fixed for simplicity
            'Tumor Location': [1],  # Fixed for simplicity
            'Prostate Boundary Distance (cm)': [1.0],  # Fixed for simplicity
            'Erectile Vessel Distance (cm)': [1.0],  # Fixed for simplicity
            'Treatment Type': [1],  # Fixed for simplicity
            'Radiation Dose Prostate (Gy)': [radiation_dose_prostate],
            'Radiation Dose Vessels (Gy)': [radiation_dose_vessels],
            'Vessel Sparing': [1],  # Fixed for simplicity
        })

        # Preprocess the input data
        input_data = preprocess_data(input_data)

        # Predict using logistic regression and random forest
        pred_log_reg = log_reg.predict(input_data)[0]
        pred_rf = rf_model.predict(input_data)[0]

        # Display predictions
        st.write(f"**Logistic Regression Prediction: {'Erectile Function Preserved' if pred_log_reg == 1 else 'Erectile Function Lost'}**")
        st.write(f"**Random Forest Prediction: {'Erectile Function Preserved' if pred_rf == 1 else 'Erectile Function Lost'}**")

    # Show model performance
    st.write("## Model Performance")
    y_pred_log_reg = log_reg.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)

    st.write("### Logistic Regression Report")
    st.text(classification_report(y_test, y_pred_log_reg))

    st.write("### Random Forest Report")
    st.text(classification_report(y_test, y_pred_rf))

    # Visualizations
    st.write("## Data Visualizations")
    st.write("### Correlation Heatmap")
    
    # Create a figure for the heatmap and pass it to st.pyplot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

if __name__ == '__main__':
    main()
