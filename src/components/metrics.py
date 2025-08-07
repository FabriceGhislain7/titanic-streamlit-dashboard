"""
src/components/metrics.py
Components for displaying dashboard metrics and KPIs
"""

import streamlit as st
import pandas as pd
from src.config import format_percentage, VALUE_MAPPINGS

# ----------------1. Main Overview Metrics (from notebook section 4.2.2 - Survival Analysis)
def create_overview_metrics(df):
    """
    Create main metrics for the dashboard
    Based on survival analysis from notebook section 4.2.2
    """
    if df is None:
        return
    
    # Basic survival calculations
    total_passengers = len(df)
    survived = df['Survived'].sum()
    died = total_passengers - survived
    survival_rate = (survived / total_passengers) * 100
    
    # 4-column layout for main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Passengers",
            value=f"{total_passengers:,}",
            help="Total number of passengers in the dataset"
        )
    
    with col2:
        st.metric(
            label="Survivors",
            value=f"{survived:,}",
            delta=f"{format_percentage(survival_rate)}",
            delta_color="normal",
            help="Number and percentage of survivors"
        )
    
    with col3:
        st.metric(
            label="Deaths",
            value=f"{died:,}",
            delta=f"{format_percentage(100-survival_rate)}",
            delta_color="inverse",
            help="Number and percentage of deaths"
        )
    
    with col4:
        # ----------------2. Average Age (from notebook section 4.2.1 - Age Analysis)
        avg_age = df['Age'].mean()
        st.metric(
            label="Average Age",
            value=f"{avg_age:.1f} years",
            help="Average age of passengers"
        )

# ----------------3. Class Metrics (from notebook section 4.2.2.2 - Survival by Class)
def create_class_metrics(df):
    """
    Survival metrics by passenger class
    From notebook section 4.2.2.2
    """
    if df is None:
        return
    
    # Calculate survival by class
    class_survival = df.groupby('Pclass').agg({
        'Survived': ['sum', 'count', 'mean']
    }).round(3)
    
    class_survival.columns = ['Survivors', 'Total', 'Survival_Rate']
    class_survival = class_survival.reset_index()
    
    st.subheader("Survival by Class")
    
    for _, row in class_survival.iterrows():
        pclass = int(row['Pclass'])
        class_name = VALUE_MAPPINGS['Pclass'][pclass]
        survival_rate = row['Survival_Rate'] * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{class_name} - Total", f"{int(row['Total'])}")
        with col2:
            st.metric(f"{class_name} - Survivors", f"{int(row['Survivors'])}")
        with col3:
            st.metric(f"{class_name} - Rate", f"{survival_rate:.1f}%")

# ----------------4. Gender Metrics (from notebook section 4.2.2.3 - Survival by Gender)
def create_gender_metrics(df):
    """
    Survival metrics by gender
    From notebook section 4.2.2.3
    """
    if df is None:
        return
    
    # Calculate survival by gender
    gender_survival = df.groupby('Sex').agg({
        'Survived': ['sum', 'count', 'mean']
    }).round(3)
    
    gender_survival.columns = ['Survivors', 'Total', 'Survival_Rate']
    gender_survival = gender_survival.reset_index()
    
    st.subheader("Survival by Gender")
    
    col1, col2 = st.columns(2)
    
    for i, (_, row) in enumerate(gender_survival.iterrows()):
        gender = row['Sex']
        gender_name = VALUE_MAPPINGS['Sex'][gender]
        survival_rate = row['Survival_Rate'] * 100
        
        with col1 if i == 0 else col2:
            st.metric(
                label=f"{gender_name}",
                value=f"{int(row['Survivors'])}/{int(row['Total'])}",
                delta=f"{survival_rate:.1f}%",
                help=f"Survivors/Total for {gender_name.lower()}"
            )

# ----------------5. Basic Statistical Metrics (from notebook section 4.1.1 - Descriptive Statistics)
def create_statistical_metrics(df):
    """
    Show main descriptive statistics
    From notebook section 4.1.1
    """
    if df is None:
        return
    
    st.subheader("Descriptive Statistics")
    
    # Main numerical variables
    age_stats = df['Age'].describe()
    fare_stats = df['Fare'].describe()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Min Age", f"{age_stats['min']:.0f} years")
        st.metric("Max Age", f"{age_stats['max']:.0f} years")
    
    with col2:
        st.metric("Median Age", f"{age_stats['50%']:.0f} years")
        st.metric("Mean Age", f"{age_stats['mean']:.1f} years")
    
    with col3:
        st.metric("Min Fare", f"${fare_stats['min']:.0f}")
        st.metric("Max Fare", f"${fare_stats['max']:.0f}")
    
    with col4:
        st.metric("Median Fare", f"${fare_stats['50%']:.0f}")
        st.metric("Mean Fare", f"${fare_stats['mean']:.0f}")