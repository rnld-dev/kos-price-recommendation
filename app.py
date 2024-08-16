import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Custom CSS untuk styling
st.markdown("""
    <style>
    .stApp {
        background-color: #dcedc8; /* Latar belakang hijau */
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
    }
    h1 {
        color: #00695c;
    }
    .description {
        font-size: 16px;
        color: #000000; /* Warna deskripsi hitam */
        margin-bottom: 20px;
    }
    .row-style > div {
        display: inline-block;
        vertical-align: top;
        width: 48%;
        padding: 10px;
    }
    .row-style .stNumberInput, .row-style .stRadio {
        width: 95%;
    }
    .row-style .stSelectbox {
        width: 95%;
    }
    
    .center-button {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Judul aplikasi
st.markdown('<h1 style="text-align: center;">Kalkulator Harga Sewa Indekos berbasis XGBoost Regression</h1>', unsafe_allow_html=True)

# Deskripsi singkat aplikasi
st.markdown('<div class="description">Aplikasi ini dirancang khusus untuk mengestimasi harga sewa kos di wilayah Surabaya. Fokus utama dari aplikasi ini adalah kos-kosan di sekitar kampus-kampus terkemuka seperti UPN, UNAIR, ITS, dan UNESA. Masukkan data kos Anda untuk mendapatkan rekomendasi harga sewa yang sesuai.</div>', unsafe_allow_html=True)

# Load model, scaler, dan encoder yang telah disimpan
with open('best_xgb_model.sav', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Definisikan kategori untuk encoding
jenis = ['Putri', 'Putra', 'Campur']
listrik = ['Tidak termasuk listrik', 'Termasuk listrik']
kampus = ['UPN', 'UNAIR Kampus Dharmawangsa B', 'ITS Kampus Manyar', 
          'UNAIR Kampus Dharmahusada A', 'UNAIR Kampus Merr C', 
          'ITS Kampus Cokroaminoto', 'ITS Kampus Sukolilo', 
          'UNESA Kampus Ketintang (Gayungan)', 'UNESA Kampus Lidah Wetan (Lakarsantri)']
yes_no = ['Yes', 'No']

# Layout untuk tipe indekos dan pilihan lainnya
st.markdown('<div class="row-style">', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        jenis_input = st.radio("Tipe Indekos:", jenis, index=0, horizontal=True)

    with col2:
        listrik_input = st.selectbox('Listrik:', listrik)

    with col1:
        km_mandi_dalam_input = st.selectbox('K Mandi Dalam:', yes_no)

    with col2:
        ac_input = st.selectbox('AC:', yes_no)

    with col1:
        wifi_input = st.selectbox('WiFi:', yes_no)

    with col2:
        kampus_input = st.selectbox('Kampus Terdekat:', kampus)

# Layout sejajar untuk jarak ke kampus dan luas kamar
st.markdown('<div class="row-style">', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        jarak_input = st.number_input('Jarak ke Kampus Terdekat (km):', min_value=0.0, value=1.25, step=0.5, format="%.2f")
    
    with col2:
        luas_kamar_input = st.number_input('Luas Kamar (m²):', min_value=0.0, value=9.0, step=0.5, format="%.2f")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("[Gunakan Gmaps untuk menentukan jarak.](https://maps.google.com)")

# Siapkan data untuk prediksi
data = pd.DataFrame({
    'Listrik': [listrik_input],
    'K Mandi Dalam': [km_mandi_dalam_input],
    'AC': [ac_input],
    'WiFi': [wifi_input],
    'Kampus Terdekat': [kampus_input],
    'Jarak ke Kampus Terdekat': [jarak_input],
    'Luas Kamar': [luas_kamar_input],
    'Jenis': [jenis_input]  # Kolom ini akan diubah menjadi dummy setelah encoding
})

# Lakukan Label Encoding pada kolom yang diinginkan
label_encoder_kampus = LabelEncoder()
label_encoder_kampus.fit(kampus)
data['Kampus Terdekat'] = label_encoder_kampus.transform(data['Kampus Terdekat'])

# Konversi kolom biner
data['Listrik'] = data['Listrik'].map({'Tidak termasuk listrik': 0, 'Termasuk listrik': 1})
data['K Mandi Dalam'] = data['K Mandi Dalam'].map({'Yes': 1, 'No': 0})
data['AC'] = data['AC'].map({'Yes': 1, 'No': 0})
data['WiFi'] = data['WiFi'].map({'Yes': 1, 'No': 0})

# Tambahkan kolom dummy untuk 'Jenis'
data = pd.get_dummies(data, columns=['Jenis'], drop_first=True, dtype=int)

# Pastikan kolom dummy sesuai dengan yang digunakan saat pelatihan
for col in ['Jenis_Putra', 'Jenis_Putri']:
    if col not in data.columns:
        data[col] = 0

# Reorder columns to match training data
expected_columns = ['Listrik', 'K Mandi Dalam', 'AC', 'WiFi', 'Kampus Terdekat', 
                     'Jarak ke Kampus Terdekat', 'Luas Kamar', 'Jenis_Putra', 'Jenis_Putri']
data = data[expected_columns]

# Terapkan standarisasi
scaled_features = scaler.transform(data[['Jarak ke Kampus Terdekat', 'Luas Kamar']])
data[['Jarak ke Kampus Terdekat', 'Luas Kamar']] = np.round(scaled_features, 5)

# Tambahkan tombol prediksi dengan CSS kelas 'center-button'
st.markdown('<div class="center-button">', unsafe_allow_html=True)
if st.button('Predict'):
    # Prediksi
    prediksi = model.predict(data)
    
    # Menampilkan hasil prediksi dengan ikon uang dan format rupiah
    st.markdown(f"## 💰 Rekomendasi Harga Sewa: **Rp {int(prediksi[0]):,}**")
st.markdown('</div>', unsafe_allow_html=True)
