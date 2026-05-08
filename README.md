# PRODIGY_DS_02 – Titanic EDA

> **Prodigy InfoTech – Data Science Internship | Task 02**

## 📌 Task Description
Perform data cleaning and exploratory data analysis (EDA) on the Titanic dataset. Explore the relationships between variables and identify patterns and trends in the data.

---

## 📂 Repository Structure
```
PRODIGY_DS_02/
│
├── titanic_eda.py         # Main EDA script
├── fig1_overview.png      # Overview charts
├── fig2_survival_rates.png# Survival rate analysis
├── fig3_fare_family.png   # Fare & family size
├── fig4_correlation.png   # Correlation heatmap
├── fig5_age_kde.png       # Age distribution by sex & survival
└── README.md
```

---

## 🔧 Libraries Used
- `pandas` – data manipulation
- `numpy` – numerical operations
- `matplotlib` – plotting
- `seaborn` – statistical visualizations
- `scikit-learn` – dataset loading

---

## 🧹 Data Cleaning Steps
- Imputed missing **Age** values using median grouped by Passenger Class & Sex
- Filled missing **Fare** with the overall median
- Filled missing **Embarked** with the mode
- Engineered new features:
  - `family_size` = sibsp + parch + 1
  - `is_alone` = 1 if travelling alone
  - `age_group` = Child / Teen / Young Adult / Adult / Senior
  - `fare_band` = Low / Mid / High / Very High

---

## 📊 Visualizations
| Figure | Description |
|--------|-------------|
| Fig 1  | Dataset overview – survival count, class, age, sex |
| Fig 2  | Survival rates by sex, class, age group, embarkation |
| Fig 3  | Fare distribution by class & survival vs family size |
| Fig 4  | Correlation heatmap of numeric features |
| Fig 5  | Age KDE curves by survival status & sex |

---

## 💡 Key Insights
- Overall survival rate: **~38%**
- **Females** had a much higher survival rate (~74%) vs males (~19%)
- **1st class** passengers survived at nearly 3× the rate of 3rd class
- **Children** had a relatively higher chance of survival
- Passengers **with small families** (2–4 members) survived more than those alone or in large groups
- **Higher fare** correlated with better survival odds

---

## ▶️ How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python titanic_eda.py
```

---

## 🔗 Connect
**Intern:** Abdul Saif  
**Internship:** Prodigy InfoTech – Data Science Track  
**Track Code:** DS | **Task:** 02

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com)
