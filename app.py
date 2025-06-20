import streamlit as st
import pandas as pd
import joblib
import numpy as np
import uuid
import ast
import requests

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: red;
        color: black;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    
    .stButton>button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load your trained model
model = joblib.load('svc.pkl')

# Load your datasets
description_df = pd.read_csv('dataset/description.csv')
diet_df = pd.read_csv('dataset/diets.csv')
precaution_df = pd.read_csv('dataset/precautions_df.csv')
medication_df = pd.read_csv('dataset/medications.csv')
workouts_df = pd.read_csv('dataset/workout_df.csv')

# Ensure correct column names
assert 'Disease' in description_df.columns, "Column 'Disease' not found in description_df"
assert 'Disease' in diet_df.columns, "Column 'Disease' not found in diet_df"
assert 'Disease' in precaution_df.columns, "Column 'Disease' not found in precaution_df"
assert 'Disease' in medication_df.columns, "Column 'Disease' not found in medication_df"
assert 'disease' in workouts_df.columns, "Column 'disease' not found in workout_df"  
assert 'Description' in description_df.columns, "Column 'Description' not found in description_df"
assert 'Diet' in diet_df.columns, "Column 'Diet' not found in diet_df"
assert 'Precaution_1' in precaution_df.columns, "Column 'Precaution_1' not found in precaution_df"
assert 'Medication' in medication_df.columns, "Column 'Medication' not found in medication_df"
assert 'workout' in workouts_df.columns, "Column 'workout' not found in workouts"

# Set up the title and description
st.title('Medicine Recommendation System')
st.write('Select your symptoms to get medicine recommendations.')

# Input for symptoms
symptom_columns = ['itching', 'skin_rash', 'nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']  # Replace with your actual symptom column names

# Searchable multiselect box for symptoms
selected_symptoms = st.multiselect(
    'Search and Select symptoms',
    options=symptom_columns,
    default=[]
)

# Check if the minimum number of symptoms is selected
if len(selected_symptoms) >= 2:
    # Convert input into binary format (0 or 1)
    symptoms_vector = np.array([1 if symptom in selected_symptoms else 0 for symptom in symptom_columns]).reshape(1, -1)

    if st.button('Get Recommendations'):
        # Predict the disease
        prediction = model.predict(symptoms_vector)
        disease = prediction[0]

        # Display the predicted disease
        st.subheader(f'Predicted Disease: ')
        st.write(disease)

        # Show related information
        st.subheader('Disease Description:')
        description = description_df[description_df['Disease'] == disease]['Description'].values
        st.write(description[0] if len(description) > 0 else "Description not available")

        st.subheader('Precautions:')
        precautions = precaution_df[precaution_df['Disease'] == disease]
        precaution_text = '\n'.join(
           [precaution for i in range(1, 5) 
           if f'Precaution_{i}' in precautions 
           and not pd.isna(precautions[f'Precaution_{i}'].values[0])
           for precaution in [precautions[f'Precaution_{i}'].values[0]]]
        )
        st.write(precaution_text if precaution_text else "Precautions not available")

        st.subheader('Medications:')
        medication = medication_df[medication_df['Disease'] == disease]['Medication'].values
        if len(medication) > 0:
            medication_list = ast.literal_eval(medication[0])
            st.write(', '.join(medication_list))
        else:
            st.write("Medications not available")

        st.subheader('Diet Recommendations:')
        diet = diet_df[diet_df['Disease'] == disease]['Diet'].values
        if len(diet) > 0:
            diet_list = ast.literal_eval(diet[0])
            st.write(', '.join(diet_list))
        else:
            st.write("Diet recommendations not available")

        st.subheader('Workout Recommendations:')
        workout_list = workouts_df[workouts_df['disease'] == disease]['workout'].values  # lowercase 'disease' and 'workout'
        if len(workout_list) > 0:
            st.write("Workout recommendations:")
            for workout in workout_list:
                workout_text = workout.split(',')
                for w in workout_text:
                    st.write(f"- {w.strip()}")
        else:
            st.write("Workout recommendations not available")
else:
    st.warning("Please select at least 2 symptoms.")