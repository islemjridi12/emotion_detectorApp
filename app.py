# Importation des bibliothèques nécessaires
import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import joblib 

# Chargement du modèle sauvegardé
pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))

# Importation des fonctions utilitaires pour le suivi des prédictions
from track_utils import add_prediction_details, view_all_prediction_details, create_emotionclf_table

# Dictionnaire d'émoticônes pour représenter chaque émotion
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "shame": "😳", "surprise": "😮"
}

# --- Fonctions du Modèle ---
def predict_emotions(docx):
    """
    Prédit l'émotion d'un texte donné.
    """
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    """
    Retourne les probabilités pour chaque émotion prédite.
    """
    results = pipe_lr.predict_proba([docx])
    return results

# --- Personnalisation de l'interface avec CSS ---
st.markdown("""
    <style>
        .main-title {
            color: #4CAF50;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        .sub-title {
            color: #2196F3;
            text-align: center;
            font-size: 24px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 10px;
        }
        .prediction-box {
            background-color: #E8F5E9;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px #ccc;
        }
        .emoji {
            font-size: 60px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- Interface principale de l'application ---
def main():
    # Titre principal
    st.markdown('<h1 class="main-title">🧠 Emotion Classifier App 🎭</h1>', unsafe_allow_html=True)
    
    # Menu de navigation (sidebar)
    menu = ["🏠 Home", "📊 Monitor", "❓ About"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_emotionclf_table()  # Créer une table pour le suivi des prédictions
    
    # --- Page Accueil ---
    if choice == "🏠 Home":
        st.markdown('<h2 class="sub-title">💬 Detect Emotion in Your Text</h2>', unsafe_allow_html=True)

        # Formulaire pour entrer un texte
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Enter your text here:", placeholder="Write something...")
            submit_text = st.form_submit_button(label="🔍 Analyze")

        # Lorsque l'utilisateur soumet le texte
        if submit_text:
            col1, col2 = st.columns(2)  # Diviser l'écran en 2 colonnes

            # Prédiction des émotions et probabilités
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            add_prediction_details(raw_text, prediction, np.max(probability), datetime.now())

            # Colonne gauche : Texte original et prédiction
            with col1:
                st.success("✅ Original Text")
                st.write(raw_text)

                st.success("🎯 Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.markdown(f'<div class="emoji">{emoji_icon}</div>', unsafe_allow_html=True)
                st.write(f"**Emotion: {prediction.capitalize()}**")
                st.write(f"**Confidence: {np.max(probability):.2f}**")

            # Colonne droite : Visualisation des probabilités
            with col2:
                st.success("📊 Prediction Probability")

                # Transformation des probabilités en DataFrame
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_).T.reset_index()
                proba_df.columns = ["Emotion", "Probability"]

                # Graphique en barres avec Plotly
                bar_chart = px.bar(
                    proba_df, x='Emotion', y='Probability',
                    color='Emotion', color_discrete_sequence=px.colors.sequential.Viridis,
                    title="Emotion Probability Distribution"
                )
                st.plotly_chart(bar_chart, use_container_width=True)

                # Graphique en camembert pour la distribution
                st.subheader("🧩 Pie Chart")
                pie_chart = px.pie(
                    proba_df, names='Emotion', values='Probability',
                    color='Emotion', color_discrete_sequence=px.colors.sequential.Inferno
                )
                st.plotly_chart(pie_chart, use_container_width=True)

    # --- Page Monitoring ---
    elif choice == "📊 Monitor":
        st.markdown('<h2 class="sub-title">📈 Monitoring Predictions</h2>', unsafe_allow_html=True)

        # Affichage des prédictions stockées
        with st.expander("📋 View All Predictions"):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Text', 'Prediction', 'Probability', 'Timestamp'])
            st.dataframe(df_emotions)

            # Graphe des fréquences des prédictions
            prediction_counts = df_emotions['Prediction'].value_counts().reset_index()
            prediction_counts.columns = ['Emotion', 'Count']
            bar_chart = px.bar(
                prediction_counts, x='Emotion', y='Count', color='Emotion',
                color_discrete_sequence=px.colors.sequential.Plasma, title="Prediction Frequency"
            )
            st.plotly_chart(bar_chart, use_container_width=True)

    # --- Page À propos ---
    elif choice == "❓ About":
        st.markdown('<h2 class="sub-title">About the App</h2>', unsafe_allow_html=True)
        st.info("""
        This **Emotion Classifier App** allows you to analyze the emotional tone of any text. 
        It uses a trained Machine Learning model to predict emotions such as **joy**, **sadness**, **fear**, and more.
        
        ### Features:
        - 🎭 Emotion prediction
        - 📊 Interactive visualizations (bar chart and pie chart)
        - 📈 Monitoring for previous predictions
        """)

# --- Lancer l'application ---
if __name__ == '__main__':
    main()
