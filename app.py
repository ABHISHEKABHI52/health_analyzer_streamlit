import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Health Analyzer", layout="wide")
plt.style.use("seaborn-v0_8")


# -----------------------------
# Utility functions
# -----------------------------
def load_health_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df


def compute_bmi(weight_kg: np.ndarray, height_cm: np.ndarray) -> np.ndarray:
    height_m = height_cm / 100.0
    return weight_kg / (height_m ** 2)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bmi"] = compute_bmi(df["weight_kg"].to_numpy(), df["height_cm"].to_numpy())
    df["calorie_balance"] = df["calories_in"] - df["calories_out"]
    df["is_active_day"] = df["steps"] >= 8000
    df["is_sleep_ok"] = df["sleep_hours"] >= 7
    df["steps_7d_avg"] = df["steps"].rolling(7).mean()
    df["resting_hr_7d_avg"] = df["resting_hr"].rolling(7).mean()
    return df


def get_overall_stats(df: pd.DataFrame) -> dict:
    stats = {}
    stats["date_range"] = f"{df['date'].min().date()} → {df['date'].max().date()}"
    stats["days"] = len(df)
    stats["bmi_avg"] = df["bmi"].mean()
    stats["steps_avg"] = df["steps"].mean()
    stats["hr_avg"] = df["resting_hr"].mean()
    stats["cal_in_avg"] = df["calories_in"].mean()
    stats["cal_out_avg"] = df["calories_out"].mean()
    stats["sleep_avg"] = df["sleep_hours"].mean()
    stats["active_days_pct"] = 100 * df["is_active_day"].mean()
    stats["sleep_ok_pct"] = 100 * df["is_sleep_ok"].mean()
    stats["surplus_days_pct"] = 100 * (df["calorie_balance"] > 0).mean()
    return stats


def generate_recommendations(stats: dict) -> list:
    recs = []

    if stats["active_days_pct"] < 60:
        recs.append("Steps goal (≥ 8000) ज्यादा दिनों में hit करने की कोशिश करो, कम से कम 60% days पर.")
    else:
        recs.append("Steps goal अच्छी तरह achieve हो रहा है, इसे maintain करो.")

    if stats["bmi_avg"] >= 25:
        recs.append("Average BMI थोड़ा high side पर है, हल्की physical activity और diet control help कर सकता है (general advice).")
    elif stats["bmi_avg"] < 18.5:
        recs.append("Average BMI low है, balanced nutrition और थोड़ी strength training helpful हो सकती है (general advice).")
    else:
        recs.append("Average BMI normal range में है, current routine ठीक लग रहा है.")

    if stats["hr_avg"] > 75:
        recs.append("Resting heart rate थोड़ा ज्यादा है; stress, sleep और hydration का ध्यान रखो, doubt हो तो doctor से consult करो.")
    if stats["sleep_ok_pct"] < 50:
        recs.append("Half से कम दिनों में 7+ hours sleep है, नींद थोड़ा improve करने की कोशिश करो.")

    if not recs:
        recs.append("Data काफी balanced दिख रहा है, इसी तरह consistency maintain करो.")
    return recs


# -----------------------------
# Matplotlib plots for Streamlit
# -----------------------------
def plot_bmi_trend(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df["date"], df["bmi"], marker="o", label="BMI")
    ax.axhline(18.5, color="orange", linestyle="--", label="Underweight/Normal")
    ax.axhline(24.9, color="green", linestyle="--", label="Normal/Overweight")
    ax.axhline(29.9, color="red", linestyle="--", label="Overweight/Obese")
    ax.set_title("BMI over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("BMI")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_steps_and_goal(df: pd.DataFrame, goal: int = 8000):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df["date"], df["steps"], label="Steps")
    ax.axhline(goal, color="red", linestyle="--", label=f"Goal: {goal} steps")
    ax.plot(df["date"], df["steps_7d_avg"], color="black", linewidth=2, label="7-day avg")
    ax.set_title("Daily steps & 7-day avg")
    ax.set_xlabel("Date")
    ax.set_ylabel("Steps")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_calories_balance(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3))
    width = 0.4
    x = np.arange(len(df))
    ax.bar(x - width / 2, df["calories_in"], width=width, label="Calories in", color="#1f77b4")
    ax.bar(x + width / 2, df["calories_out"], width=width, label="Calories out", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(df["date"].dt.strftime("%m-%d"), rotation=45)
    ax.set_title("Calories in vs out")
    ax.set_xlabel("Date")
    ax.set_ylabel("Calories")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_resting_hr(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df["date"], df["resting_hr"], marker="o", label="Resting HR")
    ax.plot(df["date"], df["resting_hr_7d_avg"], color="red", linewidth=2, label="7-day avg")
    ax.set_title("Resting heart rate")
    ax.set_xlabel("Date")
    ax.set_ylabel("Resting HR (bpm)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_sleep(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df["date"], df["sleep_hours"], color="#6a5acd")
    ax.axhline(7, color="green", linestyle="--", label="Target: 7 hrs")
    ax.set_title("Sleep duration")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hours")
    ax.legend()
    fig.tight_layout()
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.title("🩺 Health Analyzer Dashboard")
    st.write("NumPy + Pandas + Matplotlib + Streamlit based **virtual** health analyzer.")

    st.sidebar.header("Upload / Settings")
    uploaded_file = st.sidebar.file_uploader(
        "Upload health CSV (date, weight_kg, height_cm, calories_in, calories_out, steps, resting_hr, sleep_hours)",
        type=["csv"],
    )

    default_goal = st.sidebar.number_input("Daily steps goal", min_value=1000, max_value=30000, value=8000, step=500)

    if uploaded_file is None:
        st.info("Left side se demo ya apna CSV upload karo to see the analysis.")
        st.stop()

    df = load_health_data(uploaded_file)
    df = add_derived_columns(df)
    stats = get_overall_stats(df)

    # ---- Top metrics ----
    st.subheader("Overall snapshot")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg BMI", f"{stats['bmi_avg']:.1f}")
    col2.metric("Avg steps/day", f"{stats['steps_avg']:.0f}")
    col3.metric("Avg resting HR", f"{stats['hr_avg']:.0f} bpm")
    col4.metric("Avg sleep", f"{stats['sleep_avg']:.1f} hrs")

    st.caption(f"Data range: {stats['date_range']} | Days: {stats['days']}")

    # ---- Percentages row ----
    col5, col6, col7 = st.columns(3)
    col5.metric("Days ≥ steps goal", f"{stats['active_days_pct']:.1f}%")
    col6.metric("Days with ≥ 7 hrs sleep", f"{stats['sleep_ok_pct']:.1f}%")
    col7.metric("Days with calorie surplus", f"{stats['surplus_days_pct']:.1f}%")

    # ---- Charts ----
    st.subheader("Trends & patterns")

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_bmi_trend(df))
        st.pyplot(plot_sleep(df))
    with c2:
        st.pyplot(plot_steps_and_goal(df, goal=int(default_goal)))
        st.pyplot(plot_resting_hr(df))

    st.pyplot(plot_calories_balance(df))

    # ---- Raw data ----
    with st.expander("Show raw data"):
        st.dataframe(df.reset_index(drop=True))

    # ---- Recommendations ----
    st.subheader("Insights & recommendations (non-medical)")
    recs = generate_recommendations(stats)
    for r in recs:
        st.markdown(f"- {r}")


if __name__ == "__main__":
    main()
