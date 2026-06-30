# MIMIC-III Data Access and Usage Guide

## Step-by-Step Access Procedure

### Step 1 — CITI Training (≈3 hours, free)
1. Go to https://about.citiprogram.org/
2. Create account → select your institution
3. Complete: **"Data or Specimens Only Research"** course
4. Download PDF certificate → attach to IRB application

### Step 2 — PhysioNet Registration
1. Go to https://physionet.org/register/
2. Create account with institutional email
3. Complete profile (institution, role, research purpose)

### Step 3 — Apply for MIMIC-III Access
1. Go to https://physionet.org/content/mimiciii/
2. Click "Request access"
3. Upload CITI certificate
4. Read and sign the Data Use Agreement
5. Wait for approval (typically 1–5 business days)

### Step 4 — Download Required Tables Only
```bash
# After approval, use wget with your PhysioNet credentials
# Download ONLY the tables needed (do NOT download entire database)

wget -r -N -c -np \
  --user=YOUR_PHYSIONET_USERNAME \
  --ask-password \
  https://physionet.org/files/mimiciii/1.4/CHARTEVENTS.csv.gz
  
wget -r -N -c -np \
  --user=YOUR_PHYSIONET_USERNAME \
  --ask-password \
  https://physionet.org/files/mimiciii/1.4/ICUSTAYS.csv.gz

wget -r -N -c -np \
  --user=YOUR_PHYSIONET_USERNAME \
  --ask-password \
  https://physionet.org/files/mimiciii/1.4/PATIENTS.csv.gz
```

### Step 5 — Store Securely
```bash
# Create encrypted volume (Linux)
sudo apt install veracrypt
veracrypt --create /path/to/encrypted_mimic.vc \
  --volume-type=normal \
  --size=10G \
  --encryption=AES \
  --hash=SHA-512 \
  --filesystem=Ext4

# Mount and move data
veracrypt /path/to/encrypted_mimic.vc /mnt/mimic
mv CHARTEVENTS.csv.gz /mnt/mimic/
mv ICUSTAYS.csv.gz /mnt/mimic/
mv PATIENTS.csv.gz /mnt/mimic/
```

### Step 6 — Run MIMIC Loader
```bash
cd /path/to/SPQR-IoMT
python3 real_results/mimic_loader.py \
  --mimic-dir /mnt/mimic \
  --output-dir data/mimic_processed \
  --n-patients 5000
```

---

## Required Citation

All publications using MIMIC-III must cite:

```bibtex
@article{johnson2016mimic,
  title={MIMIC-III, a freely accessible critical care database},
  author={Johnson, Alistair EW and Pollard, Tom J and Shen, Lu and
          Lehman, Li-wei H and Feng, Mengling and Ghassemi, Mohammad
          and Moody, Benjamin and Szolovits, Peter and
          Celi, Leo Anthony and Mark, Roger G},
  journal={Scientific Data},
  volume={3},
  pages={160035},
  year={2016},
  publisher={Nature Publishing Group}
}
```

And acknowledge PhysioNet:
```
Data used in this study were obtained from the MIMIC-III Clinical
Database (v1.4) available on PhysioNet (Goldberger et al., 2000;
Johnson et al., 2016).
```

---

## MIMIC-III ItemIDs for Vitals

These are the CHARTEVENTS ItemIDs for the 5 features used:

| Feature | ItemIDs (MetaVision) | ItemIDs (CareVue) |
|---|---|---|
| Heart rate | 220045 | 211 |
| SpO₂ | 220277 | 646, 834 |
| Respiratory rate | 220210, 224690 | 615, 618 |
| Systolic BP | 220179, 220050 | 455, 6701 |
| Temperature (°C) | 223762 | 676 |
