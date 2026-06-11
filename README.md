# MLOps Deployment

Repo ini berisi konfigurasi deployment sederhana untuk layanan MLOps berbasis `docker compose`.

## Fitur utama

- `mlflow` sebagai server tracking experiment
- `prometheus` untuk monitoring
- `grafana` untuk dashboard dan alert
- penyimpanan artifact lokal di `./artifacts`
- contoh file `testing.py` untuk menguji logging MLflow

## Struktur utama

- `docker-compose.yml` — definisi container `prometheus`, `grafana`, dan `mlflow`
- `.env.example` — contoh variabel lingkungan untuk Grafana dan MLflow
- `mlruns/` — data tracking MLflow
- `artifacts/` — storage artifact yang disajikan oleh MLflow
- `testing.py` — contoh script Python untuk mengirim param/metric/artifact ke MLflow

## Prasyarat

- Docker
- Docker Compose
- Python (opsional, untuk menjalankan `testing.py`)

## Cara menjalankan

1. Salin `.env.example` menjadi `.env`:
   ```bash
   cp .env.example .env
   ```
2. Lengkapi variabel:
   - `GF_SECURITY_ADMIN_USER`
   - `GF_SECURITY_ADMIN_PASSWORD`
   - `GF_SMTP_USER`
   - `GF_SMTP_PASSWORD`
3. Jalankan container:
   ```bash
   docker compose up -d
   ```

4. Akses layanan:
   - Grafana: `http://localhost:3000`
   - Prometheus: `http://localhost:9090`
   - MLflow: `http://localhost:5000`

## Menguji MLflow

1. Pastikan container `mlflow` berjalan.
2. Jalankan `testing.py`:
   ```bash
   python testing.py
   ```
3. Buka MLflow UI dan lihat experiment terbaru.

## Catatan

- `mlflow` menggunakan SQLite lokal dan menyimpan database di `mlflow.db`.
- Artifact disajikan dari folder `./artifacts`.
