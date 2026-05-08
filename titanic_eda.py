# ============================================================
#  PRODIGY INFOTECH – Data Science Internship
#  Task 02: Exploratory Data Analysis – Titanic Dataset
#  Author : Abdul Saif
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.datasets import fetch_openml   # gets Titanic without a CSV

# ── Style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.autolayout": True})

# ============================================================
# 1.  LOAD DATA
# ============================================================
print("=" * 55)
print("  TITANIC – EXPLORATORY DATA ANALYSIS")
print("=" * 55)

titanic = fetch_openml("titanic", version=1, as_frame=True)
df = titanic.frame.copy()

# Keep only the columns we care about
cols = ["survived", "pclass", "sex", "age", "sibsp",
        "parch", "fare", "embarked"]
df = df[cols]

# Fix dtypes
df["survived"] = df["survived"].astype(int)
df["pclass"]   = df["pclass"].astype(int)
df["age"]      = pd.to_numeric(df["age"], errors="coerce")
df["fare"]     = pd.to_numeric(df["fare"], errors="coerce")

print("\n── Raw shape:", df.shape)
print("\n── First 5 rows:\n", df.head())

# ============================================================
# 2.  DATA CLEANING
# ============================================================
print("\n\n── Missing values (before cleaning):")
print(df.isnull().sum())

# Impute age with median grouped by pclass & sex
df["age"] = df.groupby(["pclass", "sex"])["age"] \
              .transform(lambda x: x.fillna(x.median()))

# Impute fare with overall median
df["fare"].fillna(df["fare"].median(), inplace=True)

# Embarked: fill 2 missing with mode
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

print("\n── Missing values (after cleaning):")
print(df.isnull().sum())

# ── Derived features ────────────────────────────────────────
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"]    = (df["family_size"] == 1).astype(int)
df["age_group"]   = pd.cut(df["age"],
                            bins=[0, 12, 18, 35, 60, 100],
                            labels=["Child", "Teen",
                                    "Young Adult", "Adult", "Senior"])
df["fare_band"]   = pd.qcut(df["fare"], q=4,
                             labels=["Low", "Mid", "High", "Very High"])

print("\n── Cleaned dataset shape:", df.shape)
print("\n── Descriptive statistics:\n", df.describe())

# ============================================================
# 3.  VISUALISATIONS
# ============================================================

# ── Fig 1 · Overview (2×2) ──────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(12, 9))
fig1.suptitle("Titanic Dataset – Overview", fontsize=15, fontweight="bold")

# 3-a  Survival count
srv = df["survived"].value_counts()
axes[0, 0].bar(["Did Not Survive", "Survived"], srv.values,
               color=["#e74c3c", "#2ecc71"], edgecolor="white", width=0.5)
axes[0, 0].set_title("Survival Count")
axes[0, 0].set_ylabel("Passengers")
for bar, val in zip(axes[0, 0].patches, srv.values):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                    str(val), ha="center", fontweight="bold")

# 3-b  Passenger class distribution
pc = df["pclass"].value_counts().sort_index()
axes[0, 1].bar([f"Class {i}" for i in pc.index], pc.values,
               color=["#3498db", "#9b59b6", "#e67e22"],
               edgecolor="white", width=0.5)
axes[0, 1].set_title("Passenger Class Distribution")
axes[0, 1].set_ylabel("Count")

# 3-c  Age distribution
axes[1, 0].hist(df["age"].dropna(), bins=30,
                color="#3498db", edgecolor="white", alpha=0.85)
axes[1, 0].axvline(df["age"].median(), color="red",
                   linestyle="--", label=f"Median: {df['age'].median():.1f}")
axes[1, 0].set_title("Age Distribution")
axes[1, 0].set_xlabel("Age")
axes[1, 0].set_ylabel("Count")
axes[1, 0].legend()

# 3-d  Sex distribution
sx = df["sex"].value_counts()
axes[1, 1].pie(sx.values, labels=sx.index,
               autopct="%1.1f%%", startangle=90,
               colors=["#3498db", "#e74c3c"])
axes[1, 1].set_title("Sex Distribution")

plt.savefig("fig1_overview.png")
plt.show()
print("✔  Saved fig1_overview.png")

# ── Fig 2 · Survival by Key Variables ───────────────────────
fig2, axes = plt.subplots(2, 2, figsize=(12, 9))
fig2.suptitle("Survival Rates by Key Variables", fontsize=15, fontweight="bold")

def survival_bar(ax, col, title, palette=None):
    sr = df.groupby(col)["survived"].mean().reset_index()
    sr.columns = [col, "survival_rate"]
    colors = palette or sns.color_palette("muted", len(sr))
    bars = ax.bar(sr[col].astype(str), sr["survival_rate"],
                  color=colors, edgecolor="white", width=0.5)
    ax.set_title(title)
    ax.set_ylabel("Survival Rate")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, sr["survival_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", fontsize=9)

survival_bar(axes[0, 0], "sex",    "Survival Rate by Sex",
             ["#3498db", "#e74c3c"])
survival_bar(axes[0, 1], "pclass", "Survival Rate by Class",
             ["#2ecc71", "#f39c12", "#e74c3c"])
survival_bar(axes[1, 0], "age_group", "Survival Rate by Age Group")
survival_bar(axes[1, 1], "embarked",  "Survival Rate by Port of Embarkation")

plt.savefig("fig2_survival_rates.png")
plt.show()
print("✔  Saved fig2_survival_rates.png")

# ── Fig 3 · Fare & Family Size ──────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle("Fare & Family Size Analysis", fontsize=15, fontweight="bold")

# Fare by class (boxplot)
df.boxplot(column="fare", by="pclass", ax=axes[0],
           patch_artist=True,
           boxprops=dict(facecolor="#3498db", alpha=0.6))
axes[0].set_title("Fare Distribution by Passenger Class")
axes[0].set_xlabel("Passenger Class")
axes[0].set_ylabel("Fare (£)")
plt.sca(axes[0])
plt.title("Fare Distribution by Passenger Class")

# Family size vs survival
fam_sr = df.groupby("family_size")["survived"].mean()
axes[1].plot(fam_sr.index, fam_sr.values, marker="o",
             color="#2ecc71", linewidth=2, markersize=8)
axes[1].set_title("Survival Rate vs Family Size")
axes[1].set_xlabel("Family Size (self + relatives)")
axes[1].set_ylabel("Survival Rate")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
axes[1].set_ylim(0, 1)

plt.savefig("fig3_fare_family.png")
plt.show()
print("✔  Saved fig3_fare_family.png")

# ── Fig 4 · Correlation Heatmap ─────────────────────────────
fig4, ax = plt.subplots(figsize=(8, 6))
num_cols = ["survived", "pclass", "age", "sibsp",
            "parch", "fare", "family_size", "is_alone"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", ax=ax, linewidths=0.5,
            vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold")
plt.savefig("fig4_correlation.png")
plt.show()
print("✔  Saved fig4_correlation.png")

# ── Fig 5 · Age KDE by Survival & Sex ───────────────────────
fig5, axes = plt.subplots(1, 2, figsize=(12, 5))
fig5.suptitle("Age Distribution by Survival & Sex",
              fontsize=15, fontweight="bold")

for ax, sex in zip(axes, ["male", "female"]):
    for surv, lbl, color in zip([0, 1],
                                 ["Did Not Survive", "Survived"],
                                 ["#e74c3c", "#2ecc71"]):
        subset = df[(df["sex"] == sex) & (df["survived"] == surv)]["age"]
        subset.plot.kde(ax=ax, label=lbl, color=color, linewidth=2)
    ax.set_title(f"{sex.capitalize()} Passengers")
    ax.set_xlabel("Age")
    ax.legend()

plt.savefig("fig5_age_kde.png")
plt.show()
print("✔  Saved fig5_age_kde.png")

# ============================================================
# 4.  KEY INSIGHTS (printed summary)
# ============================================================
print("\n" + "=" * 55)
print("  KEY INSIGHTS")
print("=" * 55)

overall_sr = df["survived"].mean()
print(f"\n• Overall survival rate        : {overall_sr:.1%}")

for sex in ["female", "male"]:
    sr = df[df["sex"] == sex]["survived"].mean()
    print(f"• Survival rate ({sex:6s})       : {sr:.1%}")

for cls in [1, 2, 3]:
    sr = df[df["pclass"] == cls]["survived"].mean()
    print(f"• Survival rate (Class {cls})       : {sr:.1%}")

alone_sr  = df[df["is_alone"]  == 1]["survived"].mean()
family_sr = df[df["is_alone"]  == 0]["survived"].mean()
print(f"• Survival rate (alone)        : {alone_sr:.1%}")
print(f"• Survival rate (with family)  : {family_sr:.1%}")

print(f"\n• Median age (survivors)       : "
      f"{df[df['survived']==1]['age'].median():.1f}")
print(f"• Median age (non-survivors)   : "
      f"{df[df['survived']==0]['age'].median():.1f}")

print("\n── Analysis complete. All figures saved as PNG files.")
