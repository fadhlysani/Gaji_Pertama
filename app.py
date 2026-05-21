import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Konfigurasi halaman (Harus menjadi perintah Streamlit pertama)
st.set_page_config(page_title="Prediksi Gaji Pertama Peserta Vokasi", layout="centered")

# 2. Load the exported assets
@st.cache_resource
def load_assets():
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('linear_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return encoders, scaler, model

try:
    loaded_encoders, loaded_scaler, loaded_model = load_assets()

    st.title("🚀 Aplikasi Prediksi Gaji Pertama")
    st.markdown("Masukkan data peserta di bawah ini untuk melihat estimasi gaji pertama (dalam Juta).")

    # 3. UI for User Input
    col1, col2 = st.columns(2)

    with col1:
        jenis_kelamin = st.selectbox("Jenis Kelamin", ['L', 'P'])
        usia = st.number_input("Usia", min_value=17, max_value=50, value=25)
        pendidikan = st.selectbox("Pendidikan", ['SMA', 'SMK', 'D3', 'S1'])
        jurusan = st.selectbox("Jurusan", ['administrasi', 'teknik las', 'desain grafis', 'teknik listrik', 'otomotif'])

    with col2:
        durasi_jam = st.number_input("Durasi Jam Pelatihan", min_value=0, max_value=100, value=60)
        nilai_ujian = st.number_input("Nilai Ujian", min_value=0.0, max_value=100.0, value=85.0)
        status_bekerja = st.selectbox("Status Bekerja", ['Belum Bekerja', 'Sudah Bekerja'])

    # 4. Prediction Logic
    if st.button("Prediksi Gaji"):
        # Prepare raw data
        input_df = pd.DataFrame({
            'Jenis_Kelamin': [jenis_kelamin],
            'Usia': [usia],
            'Pendidikan': [pendidikan],
            'Jurusan': [jurusan],
            'Durasi_Jam': [durasi_jam],
            'Nilai_Ujian': [nilai_ujian],
            'Status_Bekerja': [status_bekerja]
        })

        # Apply Label Encoding
        for col, le in loaded_encoders.items():
            input_df[col] = le.transform(input_df[col])

        # Apply Scaling
        input_scaled = loaded_scaler.transform(input_df)

        # Predict
        prediction = loaded_model.predict(input_scaled)

        # Display Result
        st.success(f"### Estimasi Gaji Pertama: Rp {prediction[0]:.2f} Juta")

except FileNotFoundError:
    st.error("File model, scaler, atau encoder tidak ditemukan. Pastikan file .pkl sudah diexport.")
