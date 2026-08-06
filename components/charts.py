"""
Reusable chart rendering functions. Each takes data in and calls st.pyplot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def confusion_matrix_chart(cm, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Admitted", "Admitted"], yticklabels=["Not Admitted", "Admitted"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    st.pyplot(fig)


def correlation_heatmap(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={"shrink": 0.7})
    st.pyplot(fig)


def scatter_by_outcome(df: pd.DataFrame, x: str, y: str, hue_col: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue_col,
                     palette={"Admitted": "#2a9d8f", "Not Admitted": "#e76f51"}, ax=ax)
    st.pyplot(fig)


def distribution_chart(series: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(series, kde=True, ax=ax, color="#264653")
    st.pyplot(fig)


def boxplot_by_outcome(df: pd.DataFrame, num_col: str, hue_col: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x=hue_col, y=num_col, ax=ax,
                palette={"Admitted": "#2a9d8f", "Not Admitted": "#e76f51"})
    st.pyplot(fig)


def before_after_scaling(raw_series: pd.Series, scaled_series: pd.Series, col_name: str):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Before scaling — {col_name} (train)**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(raw_series, kde=True, ax=ax, color="#e76f51")
        st.pyplot(fig)
    with c2:
        st.markdown(f"**After MinMax scaling — {col_name} (train)**")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        sns.histplot(scaled_series, kde=True, ax=ax2, color="#2a9d8f")
        st.pyplot(fig2)


def loss_curve_chart(loss_values):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(loss_values, color="#264653", label="Loss")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


def feature_importance_bar(values, index, color="#2a9d8f"):
    imp = pd.Series(values, index=index).sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, color=color)
    ax.invert_yaxis()
    st.pyplot(fig)
