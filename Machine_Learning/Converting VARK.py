import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

print('Membaca Data Excel...')
current_dir = os.getcwd()
df = pd.read_excel(os.path.join(current_dir, 'Machine_Learning','DataKuesioner.xlsx'))

print('Inisiasi Mapping...')
mappings = {
    1: {
        "Mencari tahu tokonya apakah ada di suatu tempat yang saya ketahui": 'K',
        "Tanya arah jalan kepada teman": 'A',
        "Menulis petunjuk jalan": 'R',
        "Menggunakan Peta atau Layanan Maps": 'V'
    },
    2: {
        "Melihat diagramnya": 'V',
        "Mendengarkan penjelasan orang tersebut": 'A',
        "Membaca kata-kata yang tampil di video": 'R',
        "Melakukan tindakan apa yang dilakukan oleh orang dalam video": 'K'
    },
    3: {
        "Melihat hal-hal menarik pada tur tersebut": 'K',
        "Menggunakan peta atau layanan maps dan melihat lokasi yang akan dituju": 'V',
        "Membaca detail tur pada dokumen atau rencana tur": 'R',
        "Berdiskusi dengan teman atau penyelenggara tur yang merencanakan tur tersebut": 'A'
    },
    4: {
        "Saya akan menerapkan ilmu yang saya punya pada dunia kerja nanti": 'K',
        "Saya akan bercakap dengan rekan kerja saat berdiskusi": 'A',
        "Saya akan menggunakan gambar atau video untuk menunjukkan pekerjaan saya kepada rekan kerja": 'V',
        "Saya akan menerapkan kata-kata yang baik saat mengirimkan pesan kepada rekan kerja": 'R'
    },
    5: {
        "Saya suka berdiskusi": 'A',
        "Saya suka melihat pola dari apa yang saya pelajari": 'V',
        "Saya suka mencari contoh dari apa yang saya pelajari": 'K',
        "Saya suka mencari contoh dari apa yang saya pelajari": 'R'
    },
    6: {
        "Melakukan riset dan perbandingan langsung di lokasi": 'K',
        "Membaca brosur yang berisi daftar barang yang saya cari": 'R',
        "Melihat ulasan di YouTube atau Media Sosial": 'V',
        "Berdiskusi dengan pakar": 'A'
    },
    7: {
        "Melihat teman saya bermain catur dan memperhatikannya": 'K',
        "Mendengarkan penjelasan ahli tentang bagaimana aturan permainan": 'A',
        "Melihat tutorial di YouTube": 'V',
        "Membaca instruksi tertulis dari permainan catur": 'R'
    },
    8: {
        "Memberi saya buku atau data tentang penyakit jantung yang saya alami": 'R',
        "Menggunakan alat peraga guna menjelaskan apa masalah jantung yang saya alami": 'K',
        "Saya ingin dokter menjelaskan tentang penyakit saya": 'A',
        "Menunjukkan grafik dan statistik dari penyakit jantung yang saya alami": 'V'
    },
    9: {
        "Membaca instruksi yang muncul saat saya menjalankan aplikasi tersebut": 'R',
        "Berdiskusi dengan seseorang yang paham fungsi dan kegunaan dari aplikasi tersebut": 'A',
        "Saya langsung mencoba berbagai tombol pada aplikasi tersebut": 'K',
        "Saya melihat gambar atau video yang menjelaskan tentang aplikasi tersebut": 'V'
    },
    10: {
        "Video pendek atau video panjang": 'K',
        "Interaksi pada desain yang menarik": 'V',
        "Artikel atau buku digital": 'R',
        "Radio atau Podcast": 'A'
    },
    11: {
        "Diagram tahapan proyek": 'V',
        "Laporan proyek yang sedang dikerjakan": 'R',
        "Kesempatan untuk membahas proyek dengan manajer proyek": 'A',
        "Contoh nyata terhadap proyek sejenis yang sukses": 'K'
    },
    12: {
        "Bertanya kepada seseorang mengenai fitur dan apa yang bisa kamera tersebut lakukan": 'A',
        "Membaca buku instruksi pada kotak kamera tentang bagaimana penggunaan yang benar": 'R',
        "Melihat ulasan dari seseorang yang membahas mengenai kamera tersebut": 'V',
        "Mengambil langsung beberapa foto dan membandingkannya untuk mendapat hasil terbaik": 'K'
    },
    13: {
        "Demonstrasi, model atau praktek langsung": 'K',
        "Tanya jawab dan berdiskusi": 'A',
        "Buku, artikel, novel atau bahan bacaan": 'R',
        "Grafik, diagram, peta atau video": 'V'
    },
    14: {
        "Mencontohkan dari apa yang saya lakukan": 'K',
        "Mencatat hasil yang saya peroleh": 'R',
        "Seseorang yang mengatakan umpan balik langsung kepada saya": 'A',
        "Menggunakan diagram atau video hasil dari apa yang saya lakukan": 'V'
    },
    15: {
        "Menonton video atau rekaman ulasan dari seseorang yang mengunggahnya": 'K',
        "Berdiskusi dengan pemilik langsung di lokasi": 'A',
        "Membaca deskripsi mengenai fitur dan fasilitas yang diberikan": 'R',
        "Menggunakan layanan maps untuk mencari lokasi": 'V'
    },
    16: {
        "Video tutorial yang menunjukkan setiap tahapan perakitan": 'V',
        "Saran dari seseorang yang berhasil melakukannya": 'A',
        "Petunjuk tertulis yang disertakan dalam paket pembelian": 'R',
        "Mencoba merakitnya langsung secara perlahan": 'K'
    }
}

print('Inisiasi Objek Mapping...')
result_dict = {
    'Inisial': [],
    'Visual': [],
    'Auditory': [],
    'Read/Write': [],
    'Kinesthetic': []
}

print('Memproses setiap kolom...')
for index, row in df.iterrows():
    # Membuat inisial otomatis (A1, A2, ..., An)
    inisial = f'A{index + 1}'
    
    v_count = 0
    a_count = 0
    r_count = 0
    k_count = 0
    
    print(f'Memproses kolom... {index + 1}...')
    for q_num in range(1, 17):
        q_col = df.columns[q_num]
        answer = row[q_col]
        
        # Mengklasifikasi jawaban dengan pertanyaan
        if q_num in mappings and answer in mappings[q_num]:
            category = mappings[q_num][answer]
            if category == 'V':
                v_count += 1
            elif category == 'A':
                a_count += 1
            elif category == 'R':
                r_count += 1
            elif category == 'K':
                k_count += 1

    # Menambahkan hasil ke dictionary
    result_dict['Inisial'].append(inisial)
    result_dict['Visual'].append(v_count)
    result_dict['Auditory'].append(a_count)
    result_dict['Read/Write'].append(r_count)
    result_dict['Kinesthetic'].append(k_count)

print('Membuat DataFrame dari Result...')
result_df = pd.DataFrame(result_dict)

print('Menghitung Persentase dari Result...')
result_percentage_dict = {
    'Inisial': result_dict['Inisial'],
    'Visual': [(v / 16) * 100 for v in result_dict['Visual']],
    'Auditory': [(a / 16) * 100 for a in result_dict['Auditory']],
    'Read/Write': [(r / 16) * 100 for r in result_dict['Read/Write']],
    'Kinesthetic': [(k / 16) * 100 for k in result_dict['Kinesthetic']]
}

print('Membuat DataFrame dari Result Percentage...')
result_percentage_df = pd.DataFrame(result_percentage_dict)

print('Menentukan pengkategorian VARK...')
def determine_vark_result(row):
    percentages = {
        'Visual': row['Visual'],
        'Auditory': row['Auditory'],
        'Read/Write': row['Read/Write'],
        'Kinesthetic': row['Kinesthetic']
    }
    
    # Memeriksa jika terdapat 2 atau lebih kategori yang mendominasi
    sorted_percentages = sorted(percentages.values(), reverse=True)
    if sorted_percentages[0] - sorted_percentages[1] <= 10:
        return 'Multimodal'
    
    # Menentukan 1 kategori yang mendominasi
    for category, percentage in percentages.items():
        if percentage > 40:
            return category
    
    # Jika tidak ada kategori yang mendominasi
    return 'Multimodal'

print('Menghitung hasil VARK...')
result_df['Result'] = result_percentage_df.apply(determine_vark_result, axis=1)
result_percentage_df['Result'] = result_percentage_df.apply(determine_vark_result, axis=1)

print('Menggabungkan hasil...')
df['Result'] = result_percentage_df['Result']

print('Menyimpan hasil...')
if not os.path.exists(os.path.join('Machine_Learning', 'Classification Data')):
    os.makedirs(os.path.join('Machine_Learning', 'Classification Data'))

result_df.to_excel(os.path.join('Machine_Learning', 'Classification Data', 'VARK_Count.xlsx'), index=False)
result_percentage_df.to_excel(os.path.join('Machine_Learning', 'Classification Data', 'VARK_Percentage.xlsx'), index=False)
df.to_excel(os.path.join('Machine_Learning', 'Classification Data', 'VARK_Result.xlsx'), index=False)

print(result_df)
print(result_percentage_df)
print(df)

# Hanya ambil kolom numerik untuk menghitung korelasi
numeric_df = result_df[['Visual', 'Auditory', 'Read/Write', 'Kinesthetic']]
corr_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix - Dataset VARK_Count.xlsx')
plt.show()

# Membuat Q-Q Plot untuk setiap kolom numerik
# Jika titik mendekati garis diagonal, maka data tersebut berdistribusi normal
plt.figure(figsize=(12, 8))
for i, column in enumerate(numeric_df.columns, 1):
    plt.subplot(2, 2, i)
    stats.probplot(numeric_df[column], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot for {column}')
plt.tight_layout()
plt.show()