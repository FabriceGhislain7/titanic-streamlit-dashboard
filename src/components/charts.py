"""
src/components/charts.py
Componenti per la creating di charts e visualizzazioni
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.config import COLOR_PALETTES, VALUE_MAPPINGS

# ----------------1. Grafico Overall Survival (da notebook section 4.2.2 - Survival Analysis)
def create_survival_overview_chart(df):
    """
    Crea chart a torta della survival generale
    Basato sull'analysis del notebook section 4.2.2
    """
    if df is None:
        return None
    
    # Calcola conteggi survival
    survival_counts = df['Survived'].value_counts().sort_index()
    
    # Mappa i valori alle etichette
    labels = [VALUE_MAPPINGS['Survived'][val] for val in survival_counts.index]
    values = survival_counts.values
    colors = COLOR_PALETTES['survival']
    
    # Crea chart a torta
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Survival Distribution",
        showlegend=True,
        height=400,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    return fig

# ----------------2. Class Distribution (da notebook section 4.2.2.1 - Tickets by Class)
def create_class_distribution_chart(df):
    """
    Crea chart distribution passengers per class
    Da notebook section 4.2.2.1
    """
    if df is None:
        return None
    
    # Conta passengers per class
    class_counts = df['Pclass'].value_counts().sort_index()
    
    # Mappa le classi alle etichette
    labels = [VALUE_MAPPINGS['Pclass'][val] for val in class_counts.index]
    values = class_counts.values
    colors = COLOR_PALETTES['class']
    
    # Crea chart a barre
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors),
        text=values,
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Passenger Distribution by Class",
        xaxis_title="Class",
        yaxis_title="Number of Passengers",
        height=400,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    return fig

# ----------------3. Survival by Class (da notebook section 4.2.2.2 - Survival by Class)
def create_survival_by_class_chart(df):
    """
    Grafico survival per class passengers
    Da notebook section 4.2.2.2
    """
    if df is None:
        return None
    
    # Calcola survival per class
    survival_by_class = df.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
    survival_by_class.columns = ['Deaths', 'Survivors']
    
    # Mappa le classi
    class_labels = [VALUE_MAPPINGS['Pclass'][idx] for idx in survival_by_class.index]
    
    fig = go.Figure()
    
    # Aggiungi barre per morti e survivors
    fig.add_trace(go.Bar(
        name='Deaths',
        x=class_labels,
        y=survival_by_class['Deaths'],
        marker_color=COLOR_PALETTES['survival'][0]
    ))
    
    fig.add_trace(go.Bar(
        name='Survivors',
        x=class_labels,
        y=survival_by_class['Survivors'],
        marker_color=COLOR_PALETTES['survival'][1]
    ))
    
    fig.update_layout(
        title='Survival by Class',
        xaxis_title='Class',
        yaxis_title='Number of Passengers',
        barmode='stack',
        height=400
    )
    
    return fig

# ----------------4. Distribution Eta (da notebook section 4.2.1 - Age Analysis)
def create_age_distribution_chart(df):
    """
    Istogramma distribution eta
    Da notebook section 4.2.1
    """
    if df is None:
        return None
    
    # Rimuovi missing values per l'eta
    age_data = df['Age'].dropna()
    
    fig = px.histogram(
        x=age_data,
        nbins=20,
        title="Passenger Age Distribution",
        labels={'x': 'Age (years)', 'y': 'Frequency'},
        color_discrete_sequence=[COLOR_PALETTES['primary']]
    )
    
    fig.update_layout(
        height=400,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    return fig

# ----------------5. Survival by Gender (da notebook section 4.2.2.3 - Survival by Gender)
def create_survival_by_gender_chart(df):
    """
    Grafico survival per gender
    Da notebook section 4.2.2.3
    """
    if df is None:
        return None
    
    # Calcola percentuali survival per gender
    gender_survival = df.groupby('Sex')['Survived'].mean() * 100
    
    # Mappa i generi
    gender_labels = [VALUE_MAPPINGS['Sex'][idx] for idx in gender_survival.index]
    colors = COLOR_PALETTES['gender']
    
    fig = go.Figure(data=[go.Bar(
        x=gender_labels,
        y=gender_survival.values,
        marker=dict(color=colors),
        text=[f"{val:.1f}%" for val in gender_survival.values],
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Survival Rate by Gender",
        xaxis_title="Gender",
        yaxis_title="Survival Rate (%)",
        height=400,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    return fig

# ----------------6. Grafico Combinato Dashboard (sintesi per homepage)
def create_dashboard_summary_chart(df):
    """
    Grafico riassuntivo per dashboard principale
    Combina insights chiave da multiple sezioni notebook
    """
    if df is None:
        return None
    
    # Sottocharts: Class vs Gender vs Survival
    survival_summary = df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()
    
    # Mappa etichette
    class_labels = [VALUE_MAPPINGS['Pclass'][idx] for idx in survival_summary.index]
    
    fig = go.Figure()
    
    # Aggiungi tracce per ogni gender
    for i, gender in enumerate(survival_summary.columns):
        gender_label = VALUE_MAPPINGS['Sex'][gender]
        fig.add_trace(go.Bar(
            name=gender_label,
            x=class_labels,
            y=survival_summary[gender] * 100,
            marker_color=COLOR_PALETTES['gender'][i]
        ))
    
    fig.update_layout(
        title='Survival Rate by Class and Gender',
        xaxis_title='Class',
        yaxis_title='Survival Rate (%)',
        barmode='group',
        height=400
    )
    
    return fig

# ----------------7. Heatmap Missing Values (da notebook section 2.2)
def create_missing_values_heatmap(df):
    """
    Crea heatmap dei missing values
    Da notebook section 2.2 - Missing values visualization
    """
    if df is None:
        return None
    
    # Calcola missing values per row
    missing_data = df.isnull()
    
    # Se ci sono troppi data, campiona le rows
    if len(df) > 100:
        missing_data = missing_data.sample(n=100, random_state=42)
    
    fig = go.Figure(data=go.Heatmap(
        z=missing_data.values.astype(int),
        x=missing_data.columns,
        y=list(range(len(missing_data))),
        colorscale=[[0, 'lightblue'], [1, 'red']],
        showscale=True,
        colorbar=dict(title="Missing Values", tickvals=[0, 1], ticktext=["Present", "Missing"])
    ))
    
    fig.update_layout(
        title="Heatmap Missing Values (campione 100 rows)",
        xaxis_title="Columns",
        yaxis_title="Rows (campione)",
        height=400
    )
    
    return fig

# ----------------8. Grafico Tipi di Data (da notebook - data types analysis)
def create_data_types_chart(df):
    """
    Visualizza distribution tipi di data
    """
    if df is None:
        return None
    
    data_types = df.dtypes.value_counts()
    
    # Converti in stringhe per evitare errori di serializzazione JSON
    values = data_types.values.astype(int)
    names = [str(name) for name in data_types.index]
    
    fig = px.pie(
        values=values,
        names=names,
        title="Data Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(height=400)
    return fig


