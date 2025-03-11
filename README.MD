# 🔌 Fault Detection In Electric Lines Using Anomaly Detection Models 
### BECE230E - Embedded C Programming (Winter-Sem 2024-25)  
### 🎓 22BEE1311 - Satyam Kumar  
### 📊 Data Source: Tamil Nadu Electricity Board Hourly Readings  

[Link](https://archive.ics.uci.edu/dataset/290/tamilnadu+electricity+board+hourly+readings)


---

## 📖 Project Overview  
This project focuses on **fault detection in electricity readings** using anomaly detection techniques.  
It analyzes **apparent power (VA)** and **real power (kW)** to identify unusual patterns in the data.  
By applying statistical and machine learning methods, we detect anomalies that indicate potential faults.  


---

## ⚙️ Installation & Setup  
### 1️⃣ Install Dependencies  
Ensure you have Python installed, then install required libraries:  
```sh
pip install -r requirements.txt
Or install manually:

sh
Copy
Edit
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```
2️⃣ Run the Code
Navigate to the model/ directory and run:

```sh
Copy
Edit
python main.py
```
🧑‍💻 Methodology

1️⃣ Z-Score Method

Uses standard deviation to measure how far a value deviates from the mean.
Anomalies are detected if the Z-score exceeds 3.

2️⃣ Interquartile Range (IQR) Method
Calculates Q1 (25th percentile) and Q3 (75th percentile).
Anomalies are detected if data falls beyond 1.5 × IQR.

3️⃣ Isolation Forest (ML-Based)
An unsupervised machine learning model that isolates anomalies in a dataset.

4️⃣ Local Outlier Factor (LOF)
Detects outliers by analyzing local data density.

📊 Visualization
The script generates two plots to help analyze detected anomalies.