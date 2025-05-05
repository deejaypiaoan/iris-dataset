import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset with error handling
@st.cache_data
def load_data():
    try:
        url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

df = load_data()

if df is not None:
    # Encode species labels to numbers
    le = LabelEncoder()
    df["species_encoded"] = le.fit_transform(df["species"])

    # Prepare features and target
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species_encoded"]

    # Initialize models
    linear_model = LinearRegression()
    logistic_model = LogisticRegression(max_iter=1000)
    
    # Train both models
    linear_model.fit(X, y)
    logistic_model.fit(X, y)

    with st.sidebar:
        st.title("👥 Group Members")
        st.write("""
        - Atornee O. Maala
        - Deejay D. Piaoan
        - Hener P. Lorenzana
        - Shekeina D. Dabalos
        - Joseph B. Rosales
        """)
        st.title("🔢 SUPERVISED LEARNING MODEL")
        selected_model = st.radio(
            "Select Algorithm",
            ["Linear Regression", "Logistic Regression"],
            key="model_selection"
        )
        st.image("https://miro.medium.com/v2/resize:fit:720/format:webp/1*H2UmG5L1I5bzFCW006N5Ag.png", caption="Iris Dataset")

    st.title(f"Iris Flower Species Prediction")
    st.markdown("This application uses Machine Learning models to predict the species of an Iris flower based on its dimensions.")

    # Display current model type
    st.markdown(f"### Currently using: {selected_model}")
    st.markdown("Enter the measurements of the Iris flower:")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    with col2:
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    # Automatically update prediction when inputs change
    if 'species' not in st.session_state or \
            st.session_state.sepal_length != sepal_length or \
            st.session_state.sepal_width != sepal_width or \
            st.session_state.petal_length != petal_length or \
            st.session_state.petal_width != petal_width:

        st.session_state.sepal_length = sepal_length
        st.session_state.sepal_width = sepal_width
        st.session_state.petal_length = petal_length
        st.session_state.petal_width = petal_width

        try:
            input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
            
            if selected_model == "Linear Regression":
                prediction = linear_model.predict(input_data)[0]
                predicted_class = round(prediction)
                species = le.inverse_transform([predicted_class])[0]
                confidence = 1 - abs(prediction - predicted_class)
            else:  # Logistic Regression
                prediction = logistic_model.predict(input_data)[0]
                proba = logistic_model.predict_proba(input_data)[0]
                species = le.inverse_transform([prediction])[0]
                confidence = max(proba)

            st.session_state.species = species
            st.session_state.confidence = confidence
        except:
            st.session_state.species = None
            st.session_state.confidence = None

    # Display prediction result with colored confidence level
    if 'species' in st.session_state and st.session_state.species:
        confidence_percent = st.session_state.confidence * 100
        if confidence_percent >= 80:
            color = "green"
        elif confidence_percent < 50:
            color = "red"
        else:
            color = "orange"

        st.markdown(
            f"### Predicted Species: **{st.session_state.species.capitalize()}** <span style='color:{color}'>**({confidence_percent:.2f}%)**</span>",
            unsafe_allow_html=True)
else:
    st.warning("Dataset failed to load. Please check the data source.")