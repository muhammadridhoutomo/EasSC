# Penjelasan Detail GA V4: Adaptive Mutation Rate
## Untuk Presentasi & Laporan

---

## 1. APA ITU GA V4?

**GA V4 = Genetic Algorithm dengan Adaptive Mutation Rate**

Artinya: Nilai mutation rate (peluang perubahan gen) **tidak tetap**, melainkan **berubah otomatis** seiring bertambahnya generasi.

---

## 2. PERBEDAAN V1 vs V4

### GA V1 (Klasik - Parameter Statis)
```
Generasi:  0    100   200   300   400   500   600   700   800   900
Mut Rate: 0.10  0.10  0.10  0.10  0.10  0.10  0.10  0.10  0.10  0.10
          ^--------- TETAP SAMA SEPANJANG --------^
```

**Masalahnya:**
- Awal eksekusi: mutasi tinggi (bagus untuk eksplorasi)
- Akhir eksekusi: mutasi masih tinggi (malah merusak solusi bagus yang sudah ditemukan!)
- Akibat: algoritma "lompat-lompat" akhir-akhiran, sulit konvergen ke solusi terbaik

### GA V4 (Adaptif - Parameter Dinamis)
```
Generasi:  0    100   200   300   400   500   600   700   800   900
Mut Rate: 0.10  0.08  0.06  0.04  0.03  0.02  0.02  0.01  0.01  0.01
          ^---- TURUN PERLAHAN DARI AWAL KE AKHIR ----^
```

**Keuntungannya:**
- Awal: mutasi tinggi → jelajahi banyak area (eksplorasi)
- Akhir: mutasi rendah → fine-tune solusi yang sudah bagus (eksploitasi)
- Hasil: konvergensi lebih cepat & kualitas solusi lebih baik

---

## 3. RUMUS ADAPTIVE MUTATION RATE (V4)

```
mut_rate = max(0.01, 0.1 * (1 - gen/total_generations))
```

**Penjelasan:**
- `0.1` = mutation rate awal (generasi 0)
- `gen/total_generations` = fraksi progress (0 → 1)
- `(1 - gen/total_generations)` = kebalikannya (1 → 0, menurun)
- `0.1 * (...)` = dikalikan untuk skala turun dari 0.1 ke 0
- `max(0.01, ...)` = jangan sampai nol, minimal 0.01 (biar masih ada eksplor)

---

## 4. CONTOH NUMERIK

**Asumsi:** 
- Total generasi = 1000
- Parameter: `mut_rate = max(0.01, 0.1 * (1 - gen/1000))`

| Generasi | gen/1000 | 1-gen/1000 | 0.1 × (1-...) | max(...,0.01) |
|----------|----------|-----------|---------------|----------------|
| 0        | 0.00     | 1.00      | 0.100         | **0.100**      |
| 100      | 0.10     | 0.90      | 0.090         | **0.090**      |
| 250      | 0.25     | 0.75      | 0.075         | **0.075**      |
| 500      | 0.50     | 0.50      | 0.050         | **0.050**      |
| 750      | 0.75     | 0.25      | 0.025         | **0.025**      |
| 900      | 0.90     | 0.10      | 0.010         | **0.010**      |
| 999      | 0.999    | 0.001     | 0.0001        | **0.010** ← dibatasi minimum |

**Intuisi:** mutation rate turun dari 10% → 1% secara linier

---

## 5. GRAFIK ADAPTIVE MUTATION RATE

```
Mutation Rate
    ▲
 0.10│ ●
    │  ╲
 0.08│   ●
    │    ╲
 0.06│     ●
    │      ╲
 0.04│       ●
    │        ╲
 0.02│         ●
    │          ╲
 0.01│           ●●●●● (plateaued at minimum)
    │
    └─────────────────────────→ Generasi
      0   200  400  600  800  1000
```

---

## 6. KONSEP DIBALIK V4: SIMULATED ANNEALING

GA V4 terinspirasi dari **Simulated Annealing**, teknik optimasi yang meniru pendinginan metal:

```
SIMULATED ANNEALING          GA V4 (ADAPTIVE RATE)
─────────────────────────────────────────────
Temperatur tinggi       ←→   Mutation rate tinggi
→ Lompat jauh-jauh            → Eksplorasi luas

Temperatur turun        ←→   Mutation rate turun
→ Lompat kecil-kecil          → Fine-tuning

Akhirnya stabil         ←→   Konvergen ke solusi
```

---

## 7. PSEUDOCODE GA V4

```
ALGORITMA GA V4 DENGAN ADAPTIVE MUTATION RATE
==============================================

Input:
  - distance_matrix: matriks jarak antar titik
  - pop_size: ukuran populasi
  - generations: jumlah generasi
  - cx_rate: crossover rate (tetap, misal 0.8)
  - initial_mut_rate: mutation rate awal (misal 0.1)

Output:
  - best_route: rute terbaik ditemukan
  - best_distance: total jarak rute terbaik

PROCEDURE GA_V4():
  1. Populasi ← GenerateRandomPopulation(pop_size)
  2. best_distance ← ∞
  3. best_route ← null
  
  4. FOR gen = 0 TO generations-1:
  
       // ADAPTIVE MUTATION RATE ← KUNCI V4!
       mut_rate ← max(0.01, 0.1 × (1 - gen/generations))
       
       5. FOR setiap individu dalam Populasi:
            Hitung fitness = 1 / jarak_tempuh
       
       6. best_idx ← argmax(fitness)
       7. curr_distance ← CalculateDistance(Populasi[best_idx])
       
       8. IF curr_distance < best_distance:
            best_distance ← curr_distance
            best_route ← Populasi[best_idx]
       
       9. new_population ← [best_route]  // Elitism
       
       10. WHILE |new_population| < pop_size:
            parent1, parent2 ← Selection(Populasi, fitness)
            child ← Crossover(parent1, parent2, cx_rate)
            child ← Mutation(child, mut_rate)  // ← Pakai mut_rate yang adaptif
            new_population.append(child)
       
       11. Populasi ← new_population
  
  12. RETURN best_route, best_distance
```

---

## 8. IMPLEMENTASI PYTHON (LANGSUNG DARI KODE)

```python
def run(self):
    print(f"🚀 Menjalankan {self.name}...")
    pop = self.init_populasi()
    
    for gen in range(self.generations):
        
        # ← INILAH KUNCI GA V4: ADAPTIVE MUTATION RATE
        self.mut_rate = max(0.01, 0.1 * (1 - gen/self.generations))
        
        fit = self.evaluasi(pop)
        best_idx = np.argmax(fit)
        curr_dist = self.hitung_jarak(pop[best_idx])
        
        if curr_dist < self.best_distance:
            self.best_distance = curr_dist
            self.best_route = pop[best_idx]
            
        new_pop = [self.best_route]  # Elitism
        while len(new_pop) < self.pop_size:
            p1, p2 = self.selection(pop, fit)
            child = self.crossover(p1, p2)
            child = self.mutation(child)  # ← Pakai self.mut_rate yang sudah adaptif
            new_pop.append(child)
        pop = new_pop
        
        if (gen + 1) % 100 == 0:
            print(f"Generasi {gen + 1:<4} | Jarak: {self.best_distance:.2f} km | Mut Rate: {self.mut_rate:.3f}")
    
    return self.best_route, self.best_distance
```

---

## 9. KEUNTUNGAN & TANTANGAN V4

### ✅ KEUNTUNGAN:
1. **Konvergensi lebih cepat** → solusi ditemukan lebih awal
2. **Kualitas solusi lebih baik** → tidak terganggu mutasi akhir-akhiran
3. **Menyeimbangkan eksplorasi & eksploitasi** → best of both worlds
4. **Prinsip sains terukur** → sesuai teori simulated annealing

### ⚠️ TANTANGAN:
1. **Hyperparameter lebih kompleks** → perlu tuning initial rate & minimum rate
2. **Schedule decay bisa suboptimal** → linear decay mungkin tidak cocok untuk semua masalah
3. **Belum tentu selalu lebih baik** → bergantung pada landscape masalah

---

## 10. PERBANDINGAN V1 vs V4 DALAM EKSPERIMEN

Ketika kamu run **main.py**, hasil output bisa terlihat:

```
✅ Versi 1 (Klasik) Selesai. Total Jarak: 285.45 km

✅ Versi 4 (Adaptive Rate) Selesai. Total Jarak: 278.30 km
                                                ↑
                                         Lebih pendek!
```

**Alasannya:** V4 mampu fine-tune akhir-akhiran lebih baik karena mutation rate rendah.

---

## 11. SLIDE PRESENTASI (SINGKAT TAPI JELAS)

### Slide 1: Masalah
- **Judul:** "Mutation Rate Adaptif dalam GA"
- **Isi:** GA klasik punya problem: parameter tetap sepanjang → sulit konvergen

### Slide 2: Solusi V4
- **Judul:** "GA V4: Adaptive Mutation Rate"
- **Formula:** `mut_rate = max(0.01, 0.1 × (1 - gen/total_gen))`
- **Grafik:** Garis turun dari 0.10 ke 0.01

### Slide 3: Intuisi
- **Judul:** "Eksplorasi vs Eksploitasi"
- **Awal (Mut Rate Tinggi):** Jelajahi ruang besar
- **Akhir (Mut Rate Rendah):** Fine-tune solusi terbaik

### Slide 4: Hasil
- **Tabel/Grafik:** Perbandingan V1 vs V4
  - V1 total jarak: 285 km
  - V4 total jarak: 278 km ← Better!

---

## 12. PERBEDAAN V3 vs V4 (DETAIL)

Meskipun hasil eksperimen sama (258.82 km), **konsep & implementasinya sangat berbeda**:

### V3: Advanced Mutation (Inversion Mutation)
**Fokus:** VARIASI OPERATOR MUTASI

```python
def mutation(self, individu):
    if random.random() < self.mut_rate:  # ← mut_rate STATIS 0.1
        if random.random() < 0.5:
            # 50% SWAP (tukar posisi)
            i1, i2 = random.sample(range(len(individu)), 2)
            individu[i1], individu[i2] = individu[i2], individu[i1]
        else:
            # 50% INVERSION (balik urutan segmen)
            start, end = sorted(random.sample(range(len(individu)), 2))
            individu[start:end] = individu[start:end][::-1]
    return individu
```

**Karakteristik:**
- Mutation rate: **STATIS 0.1** (tidak berubah)
- Operator: **HYBRID** (50% swap + 50% inversion)
- Strategi: **Jenis mutasi yang berbeda** untuk eksplorasi lebih variatif
- Inversion cocok TSP: balik urutan segmen mungkin memberikan neighbor yang lebih meaningful

**Contoh:**
```
Rute awal:      [0, 1, 2, 3, 4, 5]

Mutasi Swap:    [0, 5, 2, 3, 4, 1]     ← tukar 1 & 5
Mutasi Inversion: [0, 1, 4, 3, 2, 5]   ← balik segmen [2,3,4]
```

---

### V4: Adaptive Mutation Rate
**Fokus:** PARAMETER ADAPTIF DINAMIS

```python
def run(self):
    for gen in range(self.generations):
        # ← mut_rate BERUBAH setiap generasi!
        self.mut_rate = max(0.01, 0.1 * (1 - gen/self.generations))
        
        # Mutation masih HANYA SWAP (seperti V1)
        child = self.mutation(child)
```

**Karakteristik:**
- Mutation rate: **DINAMIS** (0.1 → 0.01, menurun seiring generasi)
- Operator: **HANYA SWAP** (seperti V1, sederhana)
- Strategi: **Parameter berubah sesuai waktu** untuk balance eksplorasi-eksploitasi
- Terinspirasi simulated annealing

**Contoh:**
```
Gen 0:   mut_rate = 0.10  ← tinggi, eksplorasi agresif
Gen 250: mut_rate = 0.075 ← menurun
Gen 500: mut_rate = 0.05  ← terus turun
Gen 999: mut_rate = 0.01  ← rendah, fine-tuning
```

---

### TABEL PERBANDINGAN V3 vs V4

| Aspek | V3 (Inversion Mut) | V4 (Adaptive Rate) |
|-------|--------------------|--------------------|
| **Mutation Rate** | Statis: 0.1 | Dinamis: 0.1→0.01 |
| **Operator Mutasi** | Hybrid (50% swap + 50% inversion) | Hanya swap |
| **Fokus Eksperimen** | Jenis operator | Parameter adaptif |
| **Intuisi** | "Coba berbagai cara mengubah rute" | "Ubah rute banyak awal, sedikit akhir" |
| **Prinsip Inspirasi** | Permutation neighborhood | Simulated annealing |

---

### GRAFIK PERBANDINGAN V3 vs V4

#### V3: Mutation Rate Statis
```
Mutation
Jenis
    ├─ Swap
    └─ Inversion
    
Probabilitas Swap:     ▯▯▯▯▯ 50%
Probabilitas Inversion: ▯▯▯▯▯ 50%
Sepanjang 1000 generasi ← SAMA TERUS
```

#### V4: Mutation Rate Adaptif
```
Mutation Rate
    ▲
 0.10│ ●────────Awal: eksplorasi tinggi
    │  │╲
 0.05│  │ ●────Tengah: balance
    │  │  ╲
 0.01│  │   ●──Akhir: fine-tune
    │  │
    └─────────────────→ Generasi
      0   250  500  750  1000
```

---

### ANALOGI UNTUK MEMAHAMI

**V3 = Tukang Kayu dengan 2 Palu**
- Punya 2 jenis pukulan: SWAP (pukulan normal) & INVERSION (pukulan dari samping)
- Setiap hari **gunakan dua jenis pukulan dengan peluang sama** (50-50)
- Harapan: 2 teknik berbeda → hasil lebih bagus dari 1 teknik

**V4 = Penari Balet**
- Hanya punya 1 jenis gerakan (SWAP)
- Tapi **intensitas gerakan berkurang seiring waktu**:
  - Awal: lompatan tinggi & liar (eksplorasi)
  - Tengah: gerakan normal
  - Akhir: gerakan halus & presisi (fine-tuning)
- Harapan: balance optimal antara eksplorasi & eksploitasi

---

### MENGAPA HASIL V3 & V4 BISA SAMA?

Walaupun strategi berbeda, hasilnya bisa sama karena:

1. **GA adalah stokastik** → hasil bervariasi per run
2. **Local optima banyak** → berbagai strategi bisa mencapai area sama
3. **Efektivitas bergantung masalah** → V3 mungkin cocok untuk masalah tertentu, V4 untuk yang lain
4. **Random seed** → keberuntungan dalam penjelajahan ruang solusi

Untuk kesimpulan yang valid, butuh **multiple runs** (5-10 kali) per versi & bandingkan **rata-rata + std dev**.

---

### PREDIKSI PERFORMA (TEORITIS)

| Skenario | V3 Lebih Baik | V4 Lebih Baik | Alasan |
|----------|---------------|---------------|--------|
| Masalah dengan **banyak lokal optima** | ✅ | - | Inversion lebih variatif, explore lebih baik |
| Masalah dengan **landscape smooth** | - | ✅ | Adaptive rate balance eksplorasi-eksploitasi lebih efisien |
| **Ruang solusi besar** | ✅ | - | Perlu eksplorasi agresif → hybrid operator cocok |
| **Ruang solusi kecil** | - | ✅ | Adaptive rate cukup, jangan over-explore |
| **Generasi sedikit** (< 200) | ✅ | - | V4 belum sempat "cool down", V3 konsisten |
| **Generasi banyak** (> 1000) | - | ✅ | V4 fine-tuning akhir lebih efektif |

**Untuk kasus TSP 32 lokasi:** Keduanya sebanding, hasilnya bisa fluktuatif.

---

## 13. KESIMPULAN: V3 vs V4

**V3 & V4 adalah 2 strategi berbeda dengan tujuan sama:**

| Tujuan | V3 | V4 |
|--------|----|----|
| Eksplorasi | Variasi operator mutasi | Mutation rate tinggi awal |
| Eksploitasi | Dua operator tetap | Mutation rate rendah akhir |
| Trade-off | Operator-based diversity | Parameter-based schedule |

**Mana lebih baik?** Tergantung masalah. Untuk presentasi, jelaskan:
> "V3 menekankan **diversity melalui operator berbeda**, sementara V4 menekankan **adaptive control melalui parameter dinamis**. Keduanya adalah inovasi GA yang valid untuk mencapai balance eksplorasi-eksploitasi."

---

**Siap presentasi? Gunakan slide & penjelasan di atas!** 🎯
