Tahapan Feature Engineering
Berdasarkan proses preprocessing yang telah dilakukan, berikut adalah tahapan-tahapan Feature Engineering yang dapat diterapkan:
Pengkodean Variabel Kategorikal (One-Hot Encoding)
Tujuan: Mengubah fitur kategorikal menjadi format numerik yang dapat dipahami oleh model machine learning.
Tahapan:
Gunakan fungsi pd.get_dummies dari pandas untuk mengubah kolom jenis_kelamin dan status_merokok menjadi kolom numerik biner.
Parameter drop_first=True digunakan untuk menghindari multicollinearity dengan menghilangkan kolom pertama dari setiap fitur.
Kode:
python
data = pd.get_dummies(data, columns=['jenis_kelamin', 'status_merokok'], drop_first=True)
Penjelasan: Setelah tahap ini, kolom jenis_kelamin akan diubah menjadi jenis_kelamin_Pria (1 jika pria, 0 jika bukan), dan status_merokok menjadi status_merokok_Ya (1 jika merokok, 0 jika tidak).
Penanganan Outlier
Tujuan: Mengatasi nilai-nilai ekstrem yang dapat mempengaruhi kinerja model.
Tahapan:
Hitung Interquartile Range (IQR) untuk kolom tekanan_darah.
Tentukan batas bawah dan batas atas menggunakan rumus:
batas_bawah = Q1 - 3 * IQR
batas_atas = Q3 + 3 * IQR
Gunakan fungsi clip untuk membatasi nilai-nilai di luar batas ini.
Kode:
python
Q1 = data['tekanan_darah'].quantile(0.25)
Q3 = data['tekanan_darah'].quantile(0.75)
IQR = Q3 - Q1
batas_bawah = Q1 - 3 * IQR
batas_atas = Q3 + 3 * IQR
data['tekanan_darah'] = data['tekanan_darah'].clip(batas_bawah, batas_atas)
Penjelasan: Nilai-nilai ekstrem pada kolom tekanan_darah akan disesuaikan menjadi nilai batas yang telah ditentukan.
Penskalaan Fitur (Feature Scaling)
Tujuan: Menyeragamkan skala fitur numerik untuk mencegah fitur dengan skala besar mendominasi model.
Tahapan:
Gunakan StandardScaler untuk melakukan standardisasi pada kolom numerik.
Fitur-fitur yang diskalakan adalah usia, tekanan_darah, detak_jantung, kadar_gula, dan kolesterol.
Kode:
python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
kolom_numerik = ['usia', 'tekanan_darah', 'detak_jantung', 'kadar_gula', 'kolesterol']
data[kolom_numerik] = scaler.fit_transform(data[kolom_numerik])
Penjelasan: Setiap fitur numerik akan diubah sehingga memiliki mean 0 dan standar deviasi 1.
Pembuatan Fitur Baru (Jika Relevan)
Tujuan: Menciptakan fitur-fitur baru yang dapat meningkatkan kinerja model.
Contoh:
Membuat fitur kombinasi antara BMI dan Konsumsi_Rokok untuk melihat dampak gabungan terhadap kondisi jantung.
Mengubah fitur Frekuensi_Olahraga menjadi kategori (rendah, sedang, tinggi) berdasarkan nilai tertentu.
Kode (Contoh):
python
# Membuat fitur interaksi antara BMI dan Konsumsi_Rokok
data['interaksi_BMI_Rokok'] = data['BMI'] * data['Konsumsi_Rokok']

# Mengkategorikan Frekuensi_Olahraga
def kategorisasi_olahraga(frekuensi):
    if frekuensi <= 2:
        return 'rendah'
    elif frekuensi <= 4:
        return 'sedang'
    else:
        return 'tinggi'

data['kategori_olahraga'] = data['Frekuensi_Olahraga'].apply(kategorisasi_olahraga)
data = pd.get_dummies(data, columns=['kategori_olahraga'], drop_first=True)
