import streamlit as st
from analyzer import analyze_vod
import tempfile
import pandas as pd

st.set_page_config(page_title="VOD Analyzer", layout="wide")
st.title("🎥 Analyseur de VOD : détection automatique des meilleurs moments")

# Upload de la vidéo
video_file = st.file_uploader("📤 Dépose ta vidéo MP4 ici", type=["mp4"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    st.video(tmp_path)
    
    if st.button("🚀 Lancer l’analyse"):
        with st.spinner("Analyse en cours..."):
            top_moments = analyze_vod(tmp_path)
            st.success("Analyse terminée ✅")

            st.subheader("📊 Moments forts détectés")
            st.dataframe(top_moments)

            csv = top_moments.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Télécharger les résultats (CSV)", csv, "moments.csv", "text/csv")
