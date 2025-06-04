import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("iris.csv")  # Pastikan file iris.csv ada di direktori yang sama
    return df

df = load_data()

# Preprocessing
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("🌼 Klasifikasi Bunga Iris")

st.write("Masukkan panjang dan lebar sepal serta petal untuk memprediksi jenis bunga iris:")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Prediksi
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
predicted_species = prediction[0]

# Output
st.subheader("Hasil Prediksi")
st.success(f"Jenis bunga: **{predicted_species}**")

# Tampilkan data jika diinginkan
if st.checkbox("Lihat 10 data pertama"):
    st.dataframe(df.head(10))
