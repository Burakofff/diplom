import pandas as pd
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Загружаем модель
@st.cache_resource
def load_model():
    with open("model_v2.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

print('great!')

# Функция для предсказания
def predict(df):
    predictions = model.predict(df)
    return predictions

def to_numeric(df):
    # Словарь для Ordinal Encoding
    stress_mapping = {'Low': 0, 'Medium': 1, 'High': 2}

    # Применение кодирования
    df['Stress_Level'] = df['Stress_Level'].map(stress_mapping)
    #df['Mental_Health_Condition'] = df['Mental_Health_Condition'].fillna(0)

    # stress_mapping = {'Depression': 3, 'Anxiety': 2, 'Burnout': 1, 0: 0}
    # # Применение кодирования
    # df['Mental_Health_Condition'] = df['Mental_Health_Condition'].map(stress_mapping)

    stress_mapping = {'Yes': 1, 'No': 0}
    # Применение кодирования
    df['Access_to_Mental_Health_Resources'] = df['Access_to_Mental_Health_Resources'].map(stress_mapping).astype(int)

    mapping = {'Decrease': -1, 'No Change': 0, 'Increase': 1}
    df['Productivity_Change'] = df['Productivity_Change'].map(mapping).astype(int)

    mapping = {'Unsatisfied': -1, 'Neutral': 0, 'Satisfied': 1}
    df['Satisfaction_with_Remote_Work'] = df['Satisfaction_with_Remote_Work'].map(mapping).astype(int)

    work_location_mapping = {'Onsite': 0, 'Hybrid': 1, 'Remote': 2}
    df['Work_Location'] = df['Work_Location'].map(work_location_mapping).astype(int)

    df['Physical_Activity'] = df['Physical_Activity'].fillna(0)
    physical_activity_mapping = {0: 0, 'Weekly': 1, 'Daily': 2}
    df['Physical_Activity'] = df['Physical_Activity'].fillna(0).map(physical_activity_mapping)

    physical_activity_mapping = {'Average': 0, 'Poor': 1, 'Good': 2}
    df['Sleep_Quality'] = df['Sleep_Quality'].fillna(0).map(physical_activity_mapping)

  #  df.drop(columns=['Job_Role', 'Gender', 'Industry'], inplace=True)  # , 'Gender', 'Industry'


st.title("Predicting the mental health of remote workers")
input_method = st.radio("Choose input method:", ["Upload file", "Enter manually"])

if input_method == "Upload file":


    uploaded_file = st.file_uploader("Download CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            # Определяем тип файла
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("The file was uploaded successfully. The first rows of data:")
            st.dataframe(df.head())


            required_columns = ['Age', 'Years_of_Experience', 'Work_Location', 'Hours_Worked_Per_Week',
                                'Number_of_Virtual_Meetings', 'Work_Life_Balance_Rating',
                                'Stress_Level', 'Access_to_Mental_Health_Resources',
                                'Productivity_Change', 'Social_Isolation_Rating',
                                'Satisfaction_with_Remote_Work', 'Company_Support_for_Remote_Work',
                                'Physical_Activity', 'Sleep_Quality']

            if not all(col in df.columns for col in required_columns):
                st.error(f"Error: The file must contain columns{required_columns}")

            else:

                df_input = df[required_columns]
                to_numeric(df_input)

                # add new features
                df_input["Age_Exp_Interaction"] = df_input["Age"] * df_input["Years_of_Experience"]
                df_input["Workload"] = df_input["Hours_Worked_Per_Week"] * df_input["Number_of_Virtual_Meetings"]
                df_input["Sleep_Stress_Ratio"] = round(df_input["Sleep_Quality"] / (df_input["Stress_Level"] + 1), 2)
                df_input["Physical_Satisfaction"] = df_input["Physical_Activity"] * df_input[
                    "Satisfaction_with_Remote_Work"]

                df_input["Sleep_Stress_Ratio"] = np.where(np.isinf(df_input["Sleep_Stress_Ratio"]), 0,
                                                          df_input["Sleep_Stress_Ratio"])
                df_input["Sleep_Stress_Ratio"] = round(df_input["Sleep_Stress_Ratio"], 2)


                features = df_input.columns  # Все фичи кроме таргета
                scaler_standard = StandardScaler()
                df_input[features] = scaler_standard.fit_transform(df_input[features])


                predictions = predict(df_input)


                df["Mental_Health_Condition"] = predictions
                mask = {0: 'satisfactory', 1: 'burnout', 2: 'depression'}
                df['Mental_Health_Condition'] = df['Mental_Health_Condition'].map(mask)


                st.write("The results of the predictions:")
                st.dataframe(df[["Employee_ID", "Mental_Health_Condition"]])  # id - заменить на реальный идентификатор


                st.download_button(
                    label="Download the results in CSV file",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"File processing error: {e}")

else:

    st.subheader("Enter employee data manually")

    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
    work_location = st.selectbox("Work Location", ["Onsite", "Hybrid", "Remote"])
    hours_worked = st.number_input("Hours Worked Per Week", min_value=1, max_value=100, value=40)
    meetings = st.number_input("Number of Virtual Meetings", min_value=0, max_value=50, value=5)
    balance = st.slider("Work-Life Balance Rating", 0, 10, 5)
    stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    mental_health = st.selectbox("Access to Mental Health Resources", ["Yes", "No"])
    productivity = st.selectbox("Productivity Change", ["Decrease", "No Change", "Increase"])
    isolation = st.slider("Social Isolation Rating", 0, 10, 5)
    satisfaction = st.selectbox("Satisfaction with Remote Work", ["Unsatisfied", "Neutral", "Satisfied"])
    company_support = st.slider("Company Support for Remote Work", 0, 10, 5)
    physical_activity = st.selectbox("Physical Activity", [0, "Weekly", "Daily"])
    sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good"])

    df_manual = pd.DataFrame([{
        "Age": age, "Years_of_Experience": experience, "Work_Location": work_location,
        "Hours_Worked_Per_Week": hours_worked, "Number_of_Virtual_Meetings": meetings,
        "Work_Life_Balance_Rating": balance, "Stress_Level": stress,
        "Access_to_Mental_Health_Resources": mental_health, "Productivity_Change": productivity,
        "Social_Isolation_Rating": isolation, "Satisfaction_with_Remote_Work": satisfaction,
        "Company_Support_for_Remote_Work": company_support, "Physical_Activity": physical_activity,
        "Sleep_Quality": sleep_quality
    }])

    to_numeric(df_manual)

    df_manual["Age_Exp_Interaction"] = df_manual["Age"] * df_manual["Years_of_Experience"]
    df_manual["Workload"] = df_manual["Hours_Worked_Per_Week"] * df_manual["Number_of_Virtual_Meetings"]
    df_manual["Sleep_Stress_Ratio"] = round(df_manual["Sleep_Quality"] / (df_manual["Stress_Level"] + 1), 2)
    df_manual["Physical_Satisfaction"] = df_manual["Physical_Activity"] * df_manual["Satisfaction_with_Remote_Work"]


    prediction = predict(df_manual)[0]
    classes = {0: 'satisfactory', 1: 'burnout', 2: 'depression'}
    st.write(f"**Predicted mental health condition:** {classes[prediction]}")















































