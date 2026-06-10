import numpy as np
import random


class TabuSearch:
    """
    Tabu Search untuk Optimasi Rute Wisata Sejarah
    -----------------------------------------------
    Fitur:
    - Neighborhood campuran: Swap + 2-Opt (dipilih random tiap iterasi)
    - Adaptive Tabu Tenure: naik saat stuck, turun saat improving
    - Diversification: restart acak jika stagnan terlalu lama
    - Rating-aware fitness: prioritaskan tempat dengan rating tinggi
    - Customizable: jumlah hari (max_hari) dan kota awal (kota_awal)

    Kolom CSV yang dipakai:
    - Nama Tempat, Kota, Latitude, Longitude
    - Jam Buka, Jam Tutup, Durasi Kunjungan (menit), Rating
    """

    def __init__(self, name, df_lokasi, distance_matrix,
                 max_hari=2, kota_awal='Surabaya',
                 iterations=1000, tabu_tenure_min=10, tabu_tenure_max=40,
                 stagnation_limit=80):
        self.name = name
        self.df = df_lokasi
        self.matrix = distance_matrix
        self.max_hari = max_hari
        self.kota_awal = kota_awal
        self.iterations = iterations
        self.tabu_tenure_min = tabu_tenure_min
        self.tabu_tenure_max = tabu_tenure_max
        self.tabu_tenure = (tabu_tenure_min + tabu_tenure_max) // 2
        self.stagnation_limit = stagnation_limit
        self.jumlah_tempat = len(distance_matrix)

        self.best_route = None
        self.best_fitness = -1
        self.best_distance = float('inf')
        self.best_itinerary = []
        self.best_days = 0
        self.best_rating = 0
        self.best_visited = 0
        self.history = []

        # === Ekstrak data ke Python List untuk kecepatan ===
        self.cities = self.df['Kota'].tolist()
        self.names = self.df['Nama Tempat'].tolist()
        self.durations = self.df['Durasi Kunjungan (menit)'].astype(int).tolist()
        self.ratings = self.df['Rating'].astype(float).tolist()

        self.open_mins = []
        self.close_mins = []
        for _, row in self.df.iterrows():
            b_h, b_m = map(int, str(row['Jam Buka']).split(':'))
            t_h, t_m = map(int, str(row['Jam Tutup']).split(':'))
            self.open_mins.append(b_h * 60 + b_m)
            self.close_mins.append(t_h * 60 + t_m)

        # Daftar kota unik, urutkan supaya kota_awal di depan
        kota_semua = self.df['Kota'].unique().tolist()
        if self.kota_awal in kota_semua:
            kota_semua.remove(self.kota_awal)
            kota_semua.insert(0, self.kota_awal)
        self.urutan_kota = kota_semua

    # ============================================================
    # SOLUSI AWAL: Kota awal duluan, dalam tiap kota urut rating
    # ============================================================
    def generate_initial_solution(self):
        rute = []
        for kota in self.urutan_kota:
            # Ambil indeks lokasi di kota ini
            idx_kota = self.df[self.df['Kota'] == kota].index.tolist()
            # Urutkan berdasarkan rating tertinggi
            idx_kota.sort(key=lambda x: self.ratings[x], reverse=True)
            # Acak sedikit agar tiap restart berbeda, tapi tetap bias ke rating tinggi
            # (hanya acak 30% posisi untuk menjaga kualitas)
            n_shuffle = max(1, len(idx_kota) // 3)
            for _ in range(n_shuffle):
                i = random.randint(0, len(idx_kota) - 1)
                j = random.randint(0, len(idx_kota) - 1)
                idx_kota[i], idx_kota[j] = idx_kota[j], idx_kota[i]
            rute.extend(idx_kota)
        return rute

    # ============================================================
    # EVALUASI: Hitung itinerary dengan batas hari (max_hari)
    # ============================================================
    def hitung_itinerary(self, rute):
        current_day = 1
        current_time = 8 * 60  # mulai jam 08:00
        itinerary = []
        total_distance = 0
        total_rating = 0.0
        city_changes = 0

        current_city = self.cities[rute[0]]
        days_in_current_city = 1
        city_days = {}
        visited_count = 0

        for i in range(len(rute)):
            # Cek apakah sudah melebihi batas hari
            if current_day > self.max_hari:
                break

            idx = rute[i]
            kota = self.cities[idx]

            if i > 0:
                travel_dist = self.matrix[rute[i - 1]][idx]
                total_distance += travel_dist
                travel_time = travel_dist * 1.5  # km -> menit (estimasi)
            else:
                travel_time = 0

            # Pindah kota = hari baru
            if kota != current_city:
                city_changes += 1
                city_days[current_city] = city_days.get(current_city, 0) + days_in_current_city
                current_city = kota
                days_in_current_city = 1
                current_day += 1
                current_time = 8 * 60

                # Cek lagi setelah increment hari
                if current_day > self.max_hari:
                    break

            arrival_time = current_time + travel_time
            buka_menit = self.open_mins[idx]
            tutup_menit = self.close_mins[idx]
            durasi = self.durations[idx]

            # Tunggu kalau belum buka
            if arrival_time < buka_menit:
                arrival_time = buka_menit

            finish_time = arrival_time + durasi

            # Tidak muat di hari ini (tutup / lewat jam 20:00)
            if finish_time > tutup_menit or finish_time > 20 * 60:
                current_day += 1
                days_in_current_city += 1
                current_time = 8 * 60
                arrival_time = current_time + 30  # buffer 30 menit pagi
                if arrival_time < buka_menit:
                    arrival_time = buka_menit
                finish_time = arrival_time + durasi

                # Cek apakah hari baru masih dalam batas
                if current_day > self.max_hari:
                    break

            arr_h, arr_m = int(arrival_time // 60), int(arrival_time % 60)
            fin_h, fin_m = int(finish_time // 60), int(finish_time % 60)

            itinerary.append({
                'day': current_day,
                'city': kota,
                'place': self.names[idx],
                'arrive': f"{arr_h:02d}:{arr_m:02d}",
                'depart': f"{fin_h:02d}:{fin_m:02d}",
                'duration': durasi,
                'rating': self.ratings[idx]
            })

            total_rating += self.ratings[idx]
            visited_count += 1
            current_time = finish_time

        city_days[current_city] = city_days.get(current_city, 0) + days_in_current_city

        # === PENALTY ===
        penalty = 0
        # Penalti jika kota awal bukan yang pertama dikunjungi
        if len(itinerary) > 0 and itinerary[0]['city'] != self.kota_awal:
            penalty += 2000

        # === FITNESS: maksimalkan rating, minimalkan jarak ===
        if visited_count == 0:
            fitness = 0
        else:
            fitness = (total_rating * 500 + visited_count * 100) / (total_distance + penalty + 1)

        return fitness, itinerary, total_distance, current_day, total_rating, visited_count

    # ============================================================
    # NEIGHBORHOOD: Mixed Swap + 2-Opt
    # ============================================================
    def get_best_neighbor(self, rute, tabu_dict):
        best_neighbor = None
        best_fitness = -1
        best_move = None
        n = len(rute)

        # Pilih neighborhood secara acak: 50% swap, 50% 2-opt
        use_2opt = random.random() < 0.5

        if use_2opt:
            # === 2-OPT: reverse segmen ===
            for i in range(n - 1):
                for j in range(i + 2, n):
                    move = ('2opt', i, j)

                    new_rute = rute.copy()
                    new_rute[i:j + 1] = new_rute[i:j + 1][::-1]

                    fit, _, _, _, _, _ = self.hitung_itinerary(new_rute)

                    is_tabu = move in tabu_dict

                    if fit > best_fitness:
                        if not is_tabu or fit > self.best_fitness:
                            best_fitness = fit
                            best_neighbor = new_rute
                            best_move = move
        else:
            # === SWAP: tukar dua posisi ===
            for i in range(n):
                for j in range(i + 1, n):
                    move = ('swap', min(rute[i], rute[j]), max(rute[i], rute[j]))

                    new_rute = rute.copy()
                    new_rute[i], new_rute[j] = new_rute[j], new_rute[i]

                    fit, _, _, _, _, _ = self.hitung_itinerary(new_rute)

                    is_tabu = move in tabu_dict

                    if fit > best_fitness:
                        if not is_tabu or fit > self.best_fitness:
                            best_fitness = fit
                            best_neighbor = new_rute
                            best_move = move

        return best_neighbor, best_fitness, best_move

    # ============================================================
    # RUN: Loop utama Tabu Search
    # ============================================================
    def run(self):
        print(f"🚀 Menjalankan {self.name}...")
        print(f"   Kota Awal: {self.kota_awal} | Maks Hari: {self.max_hari}")
        print(f"   Total Lokasi: {self.jumlah_tempat} | Iterasi: {self.iterations}\n")

        # Buat solusi awal
        current = self.generate_initial_solution()
        current_fit, itin, dist, days, rating, visited = self.hitung_itinerary(current)

        self.best_route = current.copy()
        self.best_fitness = current_fit
        self.best_distance = dist
        self.best_itinerary = itin
        self.best_days = days
        self.best_rating = rating
        self.best_visited = visited

        tabu_dict = {}
        no_improve_count = 0
        restart_count = 0

        for it in range(self.iterations):
            # Cari tetangga terbaik
            neighbor, n_fit, n_move = self.get_best_neighbor(current, tabu_dict)

            if neighbor is None:
                self.history.append(self.best_distance)
                no_improve_count += 1
            else:
                current = neighbor

                # Tambah move ke tabu list
                if n_move:
                    tabu_dict[n_move] = it + self.tabu_tenure

                # Hapus move yang expired
                expired = [k for k, v in tabu_dict.items() if v <= it]
                for k in expired:
                    del tabu_dict[k]

                # Update global best
                if n_fit > self.best_fitness:
                    self.best_fitness = n_fit
                    self.best_route = current.copy()
                    _, itin, dist, days, rating, visited = self.hitung_itinerary(self.best_route)
                    self.best_distance = dist
                    self.best_itinerary = itin
                    self.best_days = days
                    self.best_rating = rating
                    self.best_visited = visited
                    no_improve_count = 0

                    # Adaptive tenure: kurangi saat improving
                    self.tabu_tenure = max(self.tabu_tenure_min, self.tabu_tenure - 2)
                else:
                    no_improve_count += 1
                    # Adaptive tenure: tambah saat stuck
                    self.tabu_tenure = min(self.tabu_tenure_max, self.tabu_tenure + 1)

            # === DIVERSIFICATION: restart jika stuck terlalu lama ===
            if no_improve_count >= self.stagnation_limit:
                restart_count += 1
                current = self.generate_initial_solution()
                no_improve_count = 0
                tabu_dict.clear()

                if restart_count <= 5:
                    print(f"      ↻  Diversification restart #{restart_count} di iterasi {it + 1}")

            self.history.append(self.best_distance)

            if (it + 1) % 100 == 0:
                print(f"      Iter {it + 1:<4} | Jarak: {self.best_distance:.2f} km | "
                      f"Hari: {self.best_days} | Dikunjungi: {self.best_visited} | "
                      f"Rating: {self.best_rating:.1f}")

        return self.best_route, self.best_distance, self.best_itinerary, self.best_days
