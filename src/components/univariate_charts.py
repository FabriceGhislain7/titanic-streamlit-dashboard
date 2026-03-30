"""
src/components/univariate_charts.py
Grafici specializzati per l'analysis univariata delle variables
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.config import COLOR_PALETTES, COLUMN_LABELS, VALUE_MAPPINGS

# ----------------1. Distribution Age Detailsata (da notebook section 4.2.1)
def create_age_distribution_detailed(df):
    """
    Crea distribution etÃ  detailsata con istogramma e KDE
    Da notebook section 4.2.1 - Age distribution
    """
    if df is None or 'Age' not in df.columns:
        return None
    
    age_data = df['Age'].dropna()
    
    fig = px.histogram(
        x=age_data,
        nbins=30,
        title="Distribution Age Passengers",
        labels={'x': 'Age (years)', 'y': 'Frequency'},
        marginal="box",  # Aggiunge boxplot sopra
        color_discrete_sequence=[COLOR_PALETTES['primary']]
    )
    
    # Aggiungi linea della media
    mean_age = age_data.mean()
    fig.add_vline(
        x=mean_age, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Mean: {mean_age:.1f} years"
    )
    
    fig.update_layout(height=500)
    return fig

# ----------------2. Analysis Numerica Completa (da notebook section 4.2.1)
def create_numerical_analysis_charts(df, variable):
    """
    Crea set completo di charts per variable numerica
    Basato sull'analysis del notebook section 4.2.1
    """
    if df is None or variable not in df.columns:
        return None
    
    data = df[variable].dropna()
    
    # Crea subplot con 4 charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Histogram - {COLUMN_LABELS.get(variable, variable)}',
            f'Boxplot - {COLUMN_LABELS.get(variable, variable)}',
            f'Q-Q Plot - {COLUMN_LABELS.get(variable, variable)}',
            f'Cumulative Distribution - {COLUMN_LABELS.get(variable, variable)}'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Istogramma
    fig.add_trace(
        go.Histogram(x=data, nbinsx=20, name="Frequency", marker_color=COLOR_PALETTES['primary']),
        row=1, col=1
    )
    
    # 2. Boxplot
    fig.add_trace(
        go.Box(y=data, name="Distribution", marker_color=COLOR_PALETTES['secondary']),
        row=1, col=2
    )
    
    # 3. Q-Q Plot (approssimato)
    sorted_data = np.sort(data)
    theoretical_quantiles = np.linspace(0, 1, len(sorted_data))
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles, 
            y=sorted_data, 
            mode='markers',
            name="Q-Q Plot",
            marker_color=COLOR_PALETTES['warning']
        ),
        row=2, col=1
    )
    
    # 4. Distribution Cumulativa
    sorted_vals = np.sort(data)
    cumulative_prob = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    fig.add_trace(
        go.Scatter(
            x=sorted_vals, 
            y=cumulative_prob, 
            mode='lines',
            name="CDF",
            line_color=COLOR_PALETTES['success']
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    return fig

# ----------------3. Analysis Categorica (da notebook section 4.2.2)
def create_categorical_analysis_chart(df, variable):
    """
    Crea chart per analysis variable categorica
    Da notebook section 4.2.2 - Categorical analysis
    """
    if df is None or variable not in df.columns:
        return None
    
    value_counts = df[variable].value_counts()
    
    # Mappa i valori se disponibile
    if variable in VALUE_MAPPINGS:
        labels = [VALUE_MAPPINGS[variable].get(val, str(val)) for val in value_counts.index]
    else:
        labels = [str(val) for val in value_counts.index]
    
    # Crea subplot con chart a barre e a torta
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Counts', 'Percentages'),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Grafico a barre
    fig.add_trace(
        go.Bar(x=labels, y=value_counts.values, name="Counts"),
        row=1, col=1
    )
    
    # Grafico a torta
    fig.add_trace(
        go.Pie(labels=labels, values=value_counts.values, name="Percentages"),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

# ----------------4. Analysis Age Completa (da notebook section 4.2.1 e 4.2.2.4)
def create_age_complete_analysis(df):
    """
    Analysis completa dell'etÃ  con multiple visualizzazioni
    Combina sezioni 4.2.1 e 4.2.2.4 del notebook
    """
    if df is None or 'Age' not in df.columns:
        return None
    
    age_data = df['Age'].dropna()
    
    # Crea subplot con 3 charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Age Distribution with KDE',
            'Boxplot for Outlier Detection',
            'Distribution by Decade',
            'Age by Gender'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Istogramma con curva KDE simulata
    fig.add_trace(
        go.Histogram(
            x=age_data, 
            nbinsx=25, 
            name="Age", 
            opacity=0.7,
            marker_color=COLOR_PALETTES['primary']
        ),
        row=1, col=1
    )
    
    # 2. Boxplot
    fig.add_trace(
        go.Box(
            y=age_data, 
            name="Distribution Age",
            marker_color=COLOR_PALETTES['secondary']
        ),
        row=1, col=2
    )
    
    # 3. Distribution per decadi
    age_decades = pd.cut(age_data, bins=range(0, 90, 10), labels=[f"{i}-{i+9}" for i in range(0, 80, 10)])
    decade_counts = age_decades.value_counts().sort_index()
    
    fig.add_trace(
        go.Bar(
            x=decade_counts.index.astype(str), 
            y=decade_counts.values,
            name="Per Decade",
            marker_color=COLOR_PALETTES['warning']
        ),
        row=2, col=1
    )
    
    # 4. Age per gender (se disponibile)
    if 'Sex' in df.columns:
        for i, sex in enumerate(df['Sex'].unique()):
            age_by_sex = df[df['Sex'] == sex]['Age'].dropna()
            sex_label = VALUE_MAPPINGS['Sex'].get(sex, sex)
            
            fig.add_trace(
                go.Box(
                    y=age_by_sex,
                    name=sex_label,
                    marker_color=COLOR_PALETTES['gender'][i]
                ),
                row=2, col=2
            )
    
    fig.update_layout(height=600, showlegend=True)
    return fig

# ----------------5. Comparison Trattamento Outliers (da notebook section 4.2.1)
def create_outlier_comparison_chart(df_original, df_processed, variable):
    """
    Confronta distribution prima e dopo trattamento outliers
    Da notebook section 4.2.1 - Outlier management
    """
    if df_original is None or df_processed is None or variable not in df_original.columns:
        return None
    
    original_data = df_original[variable].dropna()
    processed_data = df_processed[variable].dropna()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Before Handling', 'After Handling')
    )
    
    # Distribution originale
    fig.add_trace(
        go.Histogram(
            x=original_data,
            nbinsx=20,
            name="Original",
            opacity=0.7,
            marker_color=COLOR_PALETTES['danger']
        ),
        row=1, col=1
    )
    
    # Distribution processata
    fig.add_trace(
        go.Histogram(
            x=processed_data,
            nbinsx=20,
            name="Processed",
            opacity=0.7,
            marker_color=COLOR_PALETTES['success']
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        title=f"Comparison Distribution {COLUMN_LABELS.get(variable, variable)}"
    )
    
    return fig

# ----------------6. Grafico Percentili (da notebook section 4.1.1)
def create_percentiles_chart(df, variable):
    """
    Visualizza i percentili di una variable
    Da notebook section 4.1.1 - Descriptive statistics
    """
    if df is None or variable not in df.columns:
        return None
    
    data = df[variable].dropna()
    
    # Calcola percentili
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    percentile_values = [np.percentile(data, p) for p in percentiles]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=percentiles,
        y=percentile_values,
        mode='lines+markers',
        name='Percentiles',
        line=dict(color=COLOR_PALETTES['primary'], width=3),
        marker=dict(size=8)
    ))
    
    # Evidenzia quartili
    quartile_indices = [2, 3, 4]  # Q1, Q2, Q3
    quartile_values = [percentile_values[i] for i in quartile_indices]
    quartile_names = ['Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)']
    
    fig.add_trace(go.Scatter(
        x=[25, 50, 75],
        y=quartile_values,
        mode='markers',
        name='Quartiles',
        marker=dict(size=12, color=COLOR_PALETTES['secondary'])
    ))
    
    fig.update_layout(
        title=f"Percentili - {COLUMN_LABELS.get(variable, variable)}",
        xaxis_title="Percentile",
        yaxis_title="Value",
        height=400
    )
    
    return fig

# ----------------7. Distribution con Outliers Evidenziati (da notebook section 4.2.1)
def create_distribution_with_outliers(df, variable):
    """
    Show distribution con outliers evidenziati
    Da notebook section 4.2.1 - Outlier detection
    """
    if df is None or variable not in df.columns:
        return None
    
    from src.utils.data_processor import detect_outliers_iqr
    
    data = df[variable].dropna()
    outliers, lower_bound, upper_bound = detect_outliers_iqr(data)
    
    # Separa outliers e valori normali
    normal_values = data[(data >= lower_bound) & (data <= upper_bound)]
    outlier_values = data[(data < lower_bound) | (data > upper_bound)]
    
    fig = go.Figure()
    
    # Istogramma valori normali
    fig.add_trace(go.Histogram(
        x=normal_values,
        nbinsx=20,
        name="Normal Values",
        opacity=0.7,
        marker_color=COLOR_PALETTES['success']
    ))
    
    # Punti outliers
    if len(outlier_values) > 0:
        fig.add_trace(go.Scatter(
            x=outlier_values,
            y=[1] * len(outlier_values),  # Altezza fissa per visibilitÃ 
            mode='markers',
            name="Outliers",
            marker=dict(
                size=10,
                color=COLOR_PALETTES['danger'],
                symbol='diamond'
            )
        ))
    
    # Linee per i limiti
    fig.add_vline(x=lower_bound, line_dash="dash", line_color="orange", 
                  annotation_text=f"Lower Bound: {lower_bound:.2f}")
    fig.add_vline(x=upper_bound, line_dash="dash", line_color="orange",
                  annotation_text=f"Upper Bound: {upper_bound:.2f}")
    
    fig.update_layout(
        title=f"Distribution with Outliers - {COLUMN_LABELS.get(variable, variable)}",
        xaxis_title=COLUMN_LABELS.get(variable, variable),
        yaxis_title="Frequency",
        height=400
    )
    
    return fig

# ----------------8. Analysis Valore Mancante Pattern (da notebook section 2.2)
def create_missing_pattern_chart(df):
    """
    Analizza pattern dei missing values
    Da notebook section 2.2 - Missing values analysis
    """
    if df is None:
        return None
    
    # Calcola percentuali missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    missing_pct = missing_pct[missing_pct > 0]  # Solo columns con missing values
    
    if len(missing_pct) == 0:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=[COLUMN_LABELS.get(col, col) for col in missing_pct.index],
        x=missing_pct.values,
        orientation='h',
        marker_color=COLOR_PALETTES['warning'],
        text=[f"{val:.1f}%" for val in missing_pct.values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Missing Values Percentage by Variable",
        xaxis_title="Missing Percentage (%)",
        yaxis_title="Variables",
        height=400
    )
    
    return fig

# ----------------9. Distribution Multipla Comparativa (per confronti)
def create_multiple_distribution_comparison(df, variables):
    """
    Confronta distribuzioni di multiple variables numeriche
    """
    if df is None or not variables:
        return None
    
    fig = go.Figure()
    
    colors = COLOR_PALETTES['seaborn_palettes'][0:len(variables)]
    
    for i, var in enumerate(variables):
        if var in df.columns:
            data = df[var].dropna()
            
            fig.add_trace(go.Histogram(
                x=data,
                name=COLUMN_LABELS.get(var, var),
                opacity=0.6,
                nbinsx=20
            ))
    
    fig.update_layout(
        title="Multiple Distribution Comparison",
        xaxis_title="Values",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )
    
    return fig

# ----------------10. Summary Statistics Visual (da notebook section 4.1.1)
def create_summary_statistics_visual(df, variable):
    """
    Display grafica delle statistics descrittive
    Da notebook section 4.1.1
    """
    if df is None or variable not in df.columns:
        return None
    
    data = df[variable].dropna()
    
    stats = {
        'Mean': data.mean(),
        'Median': data.median(),
        'Mode': data.mode().iloc[0] if len(data.mode()) > 0 else data.median(),
        'Std. Dev.': data.std(),
        'Minimum': data.min(),
        'Maximum': data.max()
    }
    
    fig = go.Figure()
    
    # Grafico a barre delle statistics
    fig.add_trace(go.Bar(
        x=list(stats.keys()),
        y=list(stats.values()),
        marker_color=COLOR_PALETTES['primary'],
        text=[f"{val:.2f}" for val in stats.values()],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Descriptive Statistics - {COLUMN_LABELS.get(variable, variable)}",
        xaxis_title="Statistics",
        yaxis_title="Values",
        height=400
    )
    
    return fig


