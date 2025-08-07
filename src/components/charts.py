"""
src/components/charts.py
Components for creating charts and visualizations
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.config import COLOR_PALETTES, VALUE_MAPPINGS

# ----------------1. General Survival Chart (from notebook section 4.2.2 - Survival Analysis)
def create_survival_overview_chart(df):
    """
    Create general survival pie chart
    Based on notebook section 4.2.2 analysis
    """
    if df is None:
        return None
    
    # Calculate survival counts
    survival_counts = df['Survived'].value_counts().sort_index()
    
    # Map values to labels
    labels = [VALUE_MAPPINGS['Survived'][val] for val in survival_counts.index]
    values = survival_counts.values
    colors = COLOR_PALETTES['survival']
    
    # Create pie chart
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

# ----------------2. Class Distribution (from notebook section 4.2.2.1 - Tickets by Class)
def create_class_distribution_chart(df):
    """
    Create passenger distribution by class chart
    From notebook section 4.2.2.1
    """
    if df is None:
        return None
    
    # Count passengers by class
    class_counts = df['Pclass'].value_counts().sort_index()
    
    # Map classes to labels
    labels = [VALUE_MAPPINGS['Pclass'][val] for val in class_counts.index]
    values = class_counts.values
    colors = COLOR_PALETTES['class']
    
    # Create bar chart
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

# ----------------3. Survival by Class (from notebook section 4.2.2.2 - Survival by Class)
def create_survival_by_class_chart(df):
    """
    Survival by passenger class chart
    From notebook section 4.2.2.2
    """
    if df is None:
        return None
    
    # Calculate survival by class
    survival_by_class = df.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
    survival_by_class.columns = ['Died', 'Survived']
    
    # Map classes
    class_labels = [VALUE_MAPPINGS['Pclass'][idx] for idx in survival_by_class.index]
    
    fig = go.Figure()
    
    # Add bars for died and survived
    fig.add_trace(go.Bar(
        name='Died',
        x=class_labels,
        y=survival_by_class['Died'],
        marker_color=COLOR_PALETTES['survival'][0]
    ))
    
    fig.add_trace(go.Bar(
        name='Survived',
        x=class_labels,
        y=survival_by_class['Survived'],
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

# ----------------4. Age Distribution (from notebook section 4.2.1 - Age Analysis)
def create_age_distribution_chart(df):
    """
    Age distribution histogram
    From notebook section 4.2.1
    """
    if df is None:
        return None
    
    # Remove missing values for age
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

# ----------------5. Survival by Gender (from notebook section 4.2.2.3 - Survival by Gender)
def create_survival_by_gender_chart(df):
    """
    Survival by gender chart
    From notebook section 4.2.2.3
    """
    if df is None:
        return None
    
    # Calculate survival percentages by gender
    gender_survival = df.groupby('Sex')['Survived'].mean() * 100
    
    # Map genders
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

# ----------------6. Combined Dashboard Chart (summary for homepage)
def create_dashboard_summary_chart(df):
    """
    Summary chart for main dashboard
    Combines key insights from multiple notebook sections
    """
    if df is None:
        return None
    
    # Subplots: Class vs Gender vs Survival
    survival_summary = df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()
    
    # Map labels
    class_labels = [VALUE_MAPPINGS['Pclass'][idx] for idx in survival_summary.index]
    
    fig = go.Figure()
    
    # Add traces for each gender
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

# ----------------7. Missing Values Heatmap (from notebook section 2.2)
def create_missing_values_heatmap(df):
    """
    Create missing values heatmap
    From notebook section 2.2 - Missing values visualization
    """
    if df is None:
        return None
    
    # Calculate missing values per row
    missing_data = df.isnull()
    
    # If too much data, sample rows
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
        title="Missing Values Heatmap (100 rows sample)",
        xaxis_title="Columns",
        yaxis_title="Rows (sample)",
        height=400
    )
    
    return fig

# ----------------8. Data Types Chart (from notebook - data types analysis)
def create_data_types_chart(df):
    """
    Visualize data types distribution
    """
    if df is None:
        return None
    
    data_types = df.dtypes.value_counts()
    
    # Convert to strings to avoid JSON serialization errors
    values = data_types.values.astype(int)
    names = [str(name) for name in data_types.index]
    
    fig = px.pie(
        values=values,
        names=names,
        title="Data Types Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(height=400)
    return fig