"""
Data loading. The only module that touches the filesystem or uploads directly.
"""

import pandas as pd
import streamlit as st


@st.cache_data
def load_csv(file_or_path) -> pd.DataFrame:
    """Load a CSV from a path or an uploaded file-like object."""
    return pd.read_csv(file_or_path)
