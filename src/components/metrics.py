"""
src/components/metrics.py
Componenti per visualizzare metriche e KPI del dashboard
"""

import streamlit as st
import pandas as pd
from src.config import format_percentage, VALUE_MAPPINGS

# ----------------1. Metriche Overview Principali (da notebook section 4.2.2 - Survival Analysis)
def create_overview_metrics(df):
    """
    Crea le metriche principali per la dashboard
    Basato sull'analysis di survival del notebook section 4.2.2
    """
    if df is None:
        return
    
    # Calcoli base della survival
    total_passengers = len(df)
    survived = df['Survived'].sum()
    died = total_passengers - survived
    survival_rate = (survived / total_passengers) * 100
    
    # Layout a 4 columns per le metriche principali
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
        # ----------------2. Average Age (da notebook section 4.2.1 - Age Analysis)
        avg_age = df['Age'].mean()
        st.metric(
            label="Average Age",
            value=f"{avg_age:.1f} years",
            help="Average age of passengers"
        )

# ----------------3. Metriche per Class (da notebook section 4.2.2.2 - Survival by Class)
def create_class_metrics(df):
    """
    Metriche di survival per class passengers
    Da notebook section 4.2.2.2
    """
    if df is None:
        return
    
    # Calcola survival per class
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

# ----------------4. Metriche per Gender (da notebook section 4.2.2.3 - Survival by Gender)
def create_gender_metrics(df):
    """
    Metriche di survival per gender
    Da notebook section 4.2.2.3
    """
    if df is None:
        return
    
    # Calcola survival per gender
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
                help=f"Survivors/total for {gender_name.lower()}"
            )

# ----------------5. Metriche Statistics Base (da notebook section 4.1.1 - Descriptive Statistics)
def create_statistical_metrics(df):
    """
    Show statistics descrittive principali
    Da notebook section 4.1.1
    """
    if df is None:
        return
    
    st.subheader("Descriptive Statistics")
    
    # Variables numeriche principali
    age_stats = df['Age'].describe()
    fare_stats = df['Fare'].describe()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Min Age", f"{age_stats['min']:.0f} years")
        st.metric("Max Age", f"{age_stats['max']:.0f} years")
    
    with col2:
        st.metric("Median Age", f"{age_stats['50%']:.0f} years")
        st.metric("Average Age", f"{age_stats['mean']:.1f} years")
    
    with col3:
        st.metric("Min Fare", f"${fare_stats['min']:.0f}")
        st.metric("Max Fare", f"${fare_stats['max']:.0f}")
    
    with col4:
        st.metric("Median Fare", f"${fare_stats['50%']:.0f}")
        st.metric("Average Fare", f"${fare_stats['mean']:.0f}")


