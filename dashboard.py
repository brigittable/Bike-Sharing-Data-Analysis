import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
sns.set(style='dark')

# Set page config
st.set_page_config(
    page_title="Bike Sharing Dashboard",
    page_icon="🚲",
    layout="wide"
)

# ==================== Function Load Data ====================
@st.cache_data
def load_data():
    df_hour = pd.read_csv('hour.csv')
    df_day = pd.read_csv('day.csv')
    
    # Konversi dteday ke datetime
    df_hour['dteday'] = pd.to_datetime(df_hour['dteday'])
    
    # Mapping nilai musim
    musim_map = {1: 'Semi', 2: 'Panas', 3: 'Gugur', 4: 'Dingin'}
    df_hour['season'] = df_hour['season'].map(musim_map)
    
    # Mapping kondisi cuaca
    cuaca_map = {
        1: 'Cerah/Berawan Sebagian',
        2: 'Berkabut/Berawan',
        3: 'Hujan/Salju Ringan',
        4: 'Hujan/Salju Lebat'
    }
    df_hour['weathersit'] = df_hour['weathersit'].map(cuaca_map)
    
    # Menambahkan kolom user_type berdasarkan nilai 'registered' dan 'casual'
    df_hour['user_type'] = df_hour.apply(lambda row: 'Terdaftar' if row['registered'] > 0 else 'Kasual', axis=1)
    
    return df_hour

df = load_data()

# ==================== Header ====================
st.title('🚲 Bike Sharing Dashboard')
st.write('Dashboard ini menampilkan analisis pola penyewaan sepeda berdasarkan kondisi cuaca dan musim')

# ==================== Sidebar ====================
st.sidebar.header('Filter Data')

# Filter Musim
season_filter = st.sidebar.multiselect('Pilih Musim', options=df['season'].unique(), default=df['season'].unique())

# Filter Cuaca
weather_filter = st.sidebar.multiselect('Pilih Kondisi Cuaca', options=df['weathersit'].unique(), default=df['weathersit'].unique())

# Filter User Type
user_filter = st.sidebar.multiselect('Pilih Tipe Pengguna', options=df['user_type'].unique(), default=df['user_type'].unique())

# Filter Rentang Waktu
date_range = st.sidebar.date_input('Pilih Rentang Tanggal', [df['dteday'].min(), df['dteday'].max()])

# Apply filters
df_filtered = df[(df['season'].isin(season_filter)) &
                 (df['weathersit'].isin(weather_filter)) &
                 (df['user_type'].isin(user_filter)) &
                 (df['dteday'] >= pd.to_datetime(date_range[0])) &
                 (df['dteday'] <= pd.to_datetime(date_range[1]))]


# ==================== Visualisasi Penyewaan Berdasarkan Cuaca ====================
st.subheader("☁️ Pengaruh Cuaca terhadap Penyewaan")
weather_rentals = df_filtered.groupby("weathersit")["cnt"].mean().reset_index()

fig_weather = px.bar(
    weather_rentals, x="weathersit", y="cnt", 
    title="Rata-rata Penyewaan Berdasarkan Cuaca",
    labels={"cnt": "Jumlah Penyewaan", "weathersit": "Kondisi Cuaca"},
    color="weathersit",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
st.plotly_chart(fig_weather, use_container_width=True)

# ==================== Visualisasi Penyewaan Berdasarkan Musim ====================
st.subheader("🍂 Pengaruh Musim terhadap Penyewaan")
season_rentals = df_filtered.groupby("season")["cnt"].mean().reset_index()

fig_season = px.bar(
    season_rentals, x="season", y="cnt",
    title="Rata-rata Penyewaan Berdasarkan Musim",
    labels={"cnt": "Jumlah Penyewaan", "season": "Musim"},
    color="season",
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig_season, use_container_width=True)

# ==================== Proporsi Pengguna (Pie Chart) ====================
st.subheader("👥 Proporsi Tipe Pengguna")

# Menghitung total pengguna
total_rentals = df_filtered[["casual", "registered"]].sum()
user_labels = ["Casual", "Registered"]
user_colors = ["#1f77b4", "#ff7f0e"]

fig_user = px.pie(
    names=user_labels, values=total_rentals,
    title="Proporsi Pengguna Casual vs Registered",
    color=user_labels,
    color_discrete_sequence=user_colors
)
st.plotly_chart(fig_user, use_container_width=True)

# ==================== Analisis Pola Penyewaan per Jam ====================
st.subheader("⏳ Pola Penyewaan Sepeda per Jam")

hourly_rentals = df_filtered.groupby("hr")[["casual", "registered"]].mean().reset_index()

fig_hourly = px.line(
    hourly_rentals, x="hr", y=["casual", "registered"],
    labels={"value": "Jumlah Penyewaan", "hr": "Jam"},
    title="Pola Penyewaan Sepeda per Jam",
    markers=True
)
st.plotly_chart(fig_hourly, use_container_width=True)

# ==================== Analisis Pola Penyewaan per Hari ====================
st.subheader("📅 Pola Penyewaan Sepeda per Hari")

day_map = {
    0: "Minggu", 1: "Senin", 2: "Selasa", 3: "Rabu",
    4: "Kamis", 5: "Jumat", 6: "Sabtu"
}
df_filtered["weekday"] = df_filtered["weekday"].map(day_map)

daily_rentals = df_filtered.groupby("weekday")[["casual", "registered"]].mean().reset_index()

order_hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
daily_rentals = daily_rentals.set_index("weekday").reindex(order_hari).reset_index()

fig_daily = px.line(
    daily_rentals, x="weekday", y=["casual", "registered"],
    labels={"value": "Jumlah Penyewaan", "weekday": "Hari"},
    title="Pola Penyewaan Sepeda per Hari",
    markers=True
)
st.plotly_chart(fig_daily, use_container_width=True)

# ==================== Clustering Penyewaan Sepeda ====================
st.subheader("🔍 Clustering Pola Penyewaan")


# Gabungkan data berdasarkan tanggal ('dteday')
df_hour = pd.read_csv('hour.csv')
df_day = pd.read_csv('day.csv')
df_merged = df_hour.merge(df_day, on="dteday", suffixes=("_hour", "_day"))

# Pilih variabel yang digunakan untuk clustering
features = ["hr", "weekday_hour", "season_hour", "temp_hour", "hum_hour", "windspeed_hour", "cnt_hour"]
data_clustering = df_merged[features]

# Normalisasi data (pastikan sama dengan analisis sebelumnya)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_clustering)

# Menentukan jumlah cluster optimal (menggunakan hasil sebelumnya)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_merged["Cluster"] = kmeans.fit_predict(data_scaled)

# PCA untuk reduksi dimensi
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
df_merged["PCA1"] = data_pca[:, 0]
df_merged["PCA2"] = data_pca[:, 1]

# Plot Clustering dengan PCA
fig_cluster = px.scatter(
    df_merged, x="PCA1", y="PCA2", color=df_merged["Cluster"].astype(str),
    title="Visualisasi Clustering Penyewaan Sepeda",
    labels={"PCA1": "Komponen Utama 1", "PCA2": "Komponen Utama 2", "Cluster": "Kelompok"},
    color_discrete_sequence=px.colors.qualitative.Set1
)
st.plotly_chart(fig_cluster, use_container_width=True)

# ==================== Footer ====================
st.caption("Copyright © Brigittable 2025")
