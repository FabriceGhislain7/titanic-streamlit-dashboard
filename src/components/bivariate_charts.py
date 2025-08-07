"""
src/components/bivariate_charts.py
Specialized charts for bivariate analysis of survival factors
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.config import COLOR_PALETTES, COLUMN_LABELS, VALUE_MAPPINGS

# ----------------1. Detailed Survival by Class (from notebook section 4.2.2.2)
def create_survival_by_class_detailed(df):
    """
    Detailed survival by class chart
    From notebook section 4.2.2.2
    """
    if df is None:
        return None
    
    # Calculate class statistics
    class_stats = df.groupby('Pclass')['Survived'].agg(['sum', 'count', 'mean']).reset_index()
    class_stats.columns = ['Pclass', 'Survivors', 'Total', 'Rate']
    
    # Map classes
    class_labels = [VALUE_MAPPINGS['Pclass'][pclass] for pclass in class_stats['Pclass']]
    
    fig = go.Figure()
    
    # Survivors bars
    fig.add_trace(go.Bar(
        name='Survivors',
        x=class_labels,
        y=class_stats['Survivors'],
        marker_color=COLOR_PALETTES['survival'][1],
        text=class_stats['Survivors'],
        textposition='auto',
    ))
    
    # Total bars (background)
    fig.add_trace(go.Bar(
        name='Deaths',
        x=class_labels,
        y=class_stats['Total'] - class_stats['Survivors'],
        marker_color=COLOR_PALETTES['survival'][0],
        text=class_stats['Total'] - class_stats['Survivors'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Survival by Passenger Class',
        xaxis_title='Class',
        yaxis_title='Number of Passengers',
        barmode='stack',
        height=400
    )
    
    return fig

# ----------------2. Detailed Survival by Gender (from notebook section 4.2.2.3)
def create_survival_by_gender_detailed(df):
    """
    Detailed survival by gender chart
    From notebook section 4.2.2.3
    """
    if df is None:
        return None
    
    # Calculate survival percentages by gender
    gender_survival = df.groupby('Sex')['Survived'].mean() * 100
    
    # Map genders
    gender_labels = [VALUE_MAPPINGS['Sex'][sex] for sex in gender_survival.index]
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
        height=400
    )
    
    return fig

# ----------------3. Correlation Matrix (from notebook section 4.1.2)
def create_correlation_heatmap(df):
    """
    Correlation matrix heatmap
    From notebook section 4.1.2 - Spearman correlation
    """
    if df is None:
        return None
    
    # Select only numerical variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'PassengerId']
    
    if len(numeric_cols) < 2:
        return None
    
    # Calculate correlation
    corr_matrix = df[numeric_cols].corr(method='spearman')
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[COLUMN_LABELS.get(col, col) for col in corr_matrix.columns],
        y=[COLUMN_LABELS.get(col, col) for col in corr_matrix.index],
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Correlation Matrix (Spearman)",
        height=500
    )
    
    return fig

# ----------------4. Class Distribution (from notebook section 4.2.2.1)
def create_class_distribution_analysis(df):
    """
    Passenger distribution by class analysis
    From notebook section 4.2.2.1
    """
    if df is None:
        return None
    
    class_counts = df['Pclass'].value_counts().sort_index()
    class_labels = [VALUE_MAPPINGS['Pclass'][pclass] for pclass in class_counts.index]
    
    fig = go.Figure(data=[go.Bar(
        x=class_labels,
        y=class_counts.values,
        marker=dict(color=COLOR_PALETTES['class']),
        text=class_counts.values,
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Passenger Distribution by Class",
        xaxis_title="Class",
        yaxis_title="Number of Passengers",
        height=400
    )
    
    return fig

# ----------------5. Survival Rates by Class (from notebook section 4.2.2.2)
def create_survival_rates_by_class(df):
    """
    Survival rates by class
    From notebook section 4.2.2.2
    """
    if df is None:
        return None
    
    survival_rates = df.groupby('Pclass')['Survived'].mean() * 100
    class_labels = [VALUE_MAPPINGS['Pclass'][pclass] for pclass in survival_rates.index]
    
    fig = go.Figure(data=[go.Bar(
        x=class_labels,
        y=survival_rates.values,
        marker=dict(color=COLOR_PALETTES['class']),
        text=[f"{val:.1f}%" for val in survival_rates.values],
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Survival Rate by Class",
        xaxis_title="Class",
        yaxis_title="Survival Rate (%)",
        height=400
    )
    
    return fig

# ----------------6. Detailed Class Analysis (from notebook section 4.2.2.2)
def create_class_survival_detailed_analysis(df):
    """
    Detailed class analysis with survivors and deaths
    From notebook section 4.2.2.2
    """
    if df is None:
        return None
    
    # Calculate detailed statistics
    class_detail = df.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
    class_detail.columns = ['Deaths', 'Survivors']
    class_detail['Total'] = class_detail['Deaths'] + class_detail['Survivors']
    class_detail['Survival_Rate'] = (class_detail['Survivors'] / class_detail['Total']) * 100
    
    class_labels = [VALUE_MAPPINGS['Pclass'][idx] for idx in class_detail.index]
    
    # Create subplot with bars and line
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bars for survivors and deaths
    fig.add_trace(
        go.Bar(name='Deaths', x=class_labels, y=class_detail['Deaths'], 
               marker_color=COLOR_PALETTES['survival'][0]),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Bar(name='Survivors', x=class_labels, y=class_detail['Survivors'],
               marker_color=COLOR_PALETTES['survival'][1]),
        secondary_y=False,
    )
    
    # Survival rate line
    fig.add_trace(
        go.Scatter(x=class_labels, y=class_detail['Survival_Rate'],
                   mode='lines+markers', name='Survival Rate (%)',
                   line=dict(color='red', width=3), marker=dict(size=8)),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Class")
    fig.update_yaxes(title_text="Number of Passengers", secondary_y=False)
    fig.update_yaxes(title_text="Survival Rate (%)", secondary_y=True)
    fig.update_layout(title="Detailed Survival Analysis by Class", height=400)
    
    return fig

# ----------------7. Gender Survival Comparison (from notebook section 4.2.2.3)
def create_gender_survival_comparison(df):
    """
    Survival comparison between genders
    From notebook section 4.2.2.3
    """
    if df is None:
        return None
    
    gender_survival = df.groupby(['Sex', 'Survived']).size().unstack(fill_value=0)
    gender_survival.columns = ['Deaths', 'Survivors']
    
    gender_labels = [VALUE_MAPPINGS['Sex'][sex] for sex in gender_survival.index]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Deaths',
        x=gender_labels,
        y=gender_survival['Deaths'],
        marker_color=COLOR_PALETTES['survival'][0]
    ))
    
    fig.add_trace(go.Bar(
        name='Survivors',
        x=gender_labels,
        y=gender_survival['Survivors'],
        marker_color=COLOR_PALETTES['survival'][1]
    ))
    
    fig.update_layout(
        title='Survival Comparison by Gender',
        xaxis_title='Gender',
        yaxis_title='Number of Passengers',
        barmode='group',
        height=400
    )
    
    return fig

# ----------------8. Gender Distribution by Class (from notebook section 4.2.2.3)
def create_gender_class_distribution(df):
    """
    Gender distribution by class
    """
    if df is None:
        return None
    
    gender_class = df.groupby(['Pclass', 'Sex']).size().unstack(fill_value=0)
    class_labels = [VALUE_MAPPINGS['Pclass'][pclass] for pclass in gender_class.index]
    
    fig = go.Figure()
    
    for i, sex in enumerate(gender_class.columns):
        sex_label = VALUE_MAPPINGS['Sex'][sex]
        fig.add_trace(go.Bar(
            name=sex_label,
            x=class_labels,
            y=gender_class[sex],
            marker_color=COLOR_PALETTES['gender'][i]
        ))
    
    fig.update_layout(
        title='Gender Distribution by Class',
        xaxis_title='Class',
        yaxis_title='Number of Passengers',
        barmode='group',
        height=400
    )
    
    return fig

# ----------------9. Gender-Class Survival Analysis (from notebook section 4.2.2.3)
def create_gender_class_survival_analysis(df):
    """
    Survival analysis by gender and class
    From notebook section 4.2.2.3
    """
    if df is None:
        return None
    
    # Calculate survival rates by gender and class
    survival_by_gender_class = df.groupby(['Pclass', 'Sex'])['Survived'].mean() * 100
    survival_pivot = survival_by_gender_class.unstack()
    
    class_labels = [VALUE_MAPPINGS['Pclass'][pclass] for pclass in survival_pivot.index]
    
    fig = go.Figure()
    
    for i, sex in enumerate(survival_pivot.columns):
        sex_label = VALUE_MAPPINGS['Sex'][sex]
        fig.add_trace(go.Bar(
            name=sex_label,
            x=class_labels,
            y=survival_pivot[sex],
            marker_color=COLOR_PALETTES['gender'][i],
            text=[f"{val:.1f}%" for val in survival_pivot[sex]],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Survival Rate by Gender and Class',
        xaxis_title='Class',
        yaxis_title='Survival Rate (%)',
        barmode='group',
        height=400
    )
    
    return fig

# ----------------10. Age Distribution by Survival (from notebook section 4.2.2.4)
def create_age_survival_distribution(df):
    """
    Age distribution by survival
    From notebook section 4.2.2.4
    """
    if df is None or 'Age' not in df.columns:
        return None
    
    fig = go.Figure()
    
    # Histogram for survivors and deaths
    for survived, color, label in [(0, COLOR_PALETTES['survival'][0], 'Deaths'), 
                                   (1, COLOR_PALETTES['survival'][1], 'Survivors')]:
        age_data = df[df['Survived'] == survived]['Age'].dropna()
        
        fig.add_trace(go.Histogram(
            x=age_data,
            name=label,
            opacity=0.7,
            nbinsx=20,
            marker_color=color
        ))
    
    fig.update_layout(
        title='Age Distribution by Survival',
        xaxis_title='Age (years)',
        yaxis_title='Frequency',
        barmode='overlay',
        height=400
    )
    
    return fig

# ----------------11. Age Group Survival Rates (from notebook section 4.2.2.4)
def create_age_group_survival_rates(df):
    """
    Survival rates by age group
    From notebook section 4.2.2.4
    """
    if df is None or 'Age_Group' not in df.columns:
        return None
    
    age_group_survival = df.groupby('Age_Group')['Survived'].mean() * 100
    
    fig = go.Figure(data=[go.Bar(
        x=age_group_survival.index,
        y=age_group_survival.values,
        marker_color=COLOR_PALETTES['age_groups'],
        text=[f"{val:.1f}%" for val in age_group_survival.values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Survival Rate by Age Group',
        xaxis_title='Age Group',
        yaxis_title='Survival Rate (%)',
        height=400
    )
    
    return fig

# ----------------12. Age-Gender Survival Analysis (from notebook section 4.2.2.4)
def create_age_gender_survival_analysis(df):
    """
    Age-gender survival analysis
    From notebook section 4.2.2.4
    """
    if df is None or 'Age_Group' not in df.columns:
        return None
    
    # Calculate rates by age and gender
    age_gender_survival = df.groupby(['Age_Group', 'Sex'])['Survived'].mean() * 100
    survival_pivot = age_gender_survival.unstack()
    
    fig = go.Figure()
    
    for i, sex in enumerate(survival_pivot.columns):
        sex_label = VALUE_MAPPINGS['Sex'][sex]
        fig.add_trace(go.Bar(
            name=sex_label,
            x=survival_pivot.index,
            y=survival_pivot[sex],
            marker_color=COLOR_PALETTES['gender'][i]
        ))
    
    fig.update_layout(
        title='Survival Rate by Age and Gender',
        xaxis_title='Age Group',
        yaxis_title='Survival Rate (%)',
        barmode='group',
        height=400
    )
    
    return fig

# ----------------13. Survival by Fare Category (from notebook section 4.2.2.5)
def create_fare_category_survival(df):
    """
    Survival by fare category
    From notebook section 4.2.2.5
    """
    if df is None or 'Fare_Category' not in df.columns:
        return None
    
    fare_survival = df.groupby('Fare_Category')['Survived'].mean() * 100
    
    fig = go.Figure(data=[go.Bar(
        x=fare_survival.index,
        y=fare_survival.values,
        marker_color=COLOR_PALETTES['primary'],
        text=[f"{val:.1f}%" for val in fare_survival.values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Survival Rate by Fare Category',
        xaxis_title='Fare Category',
        yaxis_title='Survival Rate (%)',
        height=400
    )
    
    return fig

# ----------------14. Fare Distribution by Survival (from notebook section 4.2.2.5)
def create_fare_distribution_by_survival(df):
    """
    Fare distribution by survival
    From notebook section 4.2.2.5
    """
    if df is None:
        return None
    
    fig = go.Figure()
    
    for survived, color, label in [(0, COLOR_PALETTES['survival'][0], 'Deaths'), 
                                   (1, COLOR_PALETTES['survival'][1], 'Survivors')]:
        fare_data = df[df['Survived'] == survived]['Fare']
        
        fig.add_trace(go.Box(
            y=fare_data,
            name=label,
            marker_color=color
        ))
    
    fig.update_layout(
        title='Fare Distribution by Survival',
        yaxis_title='Ticket Fare',
        height=400
    )
    
    return fig

# ----------------15. Fare-Class-Survival Analysis
def create_fare_class_survival_analysis(df):
    """
    Combined fare-class-survival analysis
    """
    if df is None:
        return None
    
    # Average fares by class and survival
    fare_class_survival = df.groupby(['Pclass', 'Survived'])['Fare'].mean().unstack()
    fare_class_survival.columns = ['Deaths', 'Survivors']
    
    class_labels = [VALUE_MAPPINGS['Pclass'][pclass] for pclass in fare_class_survival.index]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Deaths',
        x=class_labels,
        y=fare_class_survival['Deaths'],
        marker_color=COLOR_PALETTES['survival'][0]
    ))
    
    fig.add_trace(go.Bar(
        name='Survivors',
        x=class_labels,
        y=fare_class_survival['Survivors'],
        marker_color=COLOR_PALETTES['survival'][1]
    ))
    
    fig.update_layout(
        title='Average Fare by Class and Survival',
        xaxis_title='Class',
        yaxis_title='Average Fare',
        barmode='group',
        height=400
    )
    
    return fig

# ----------------16. Survival by Family Size (from notebook section 4.2.2.6)
def create_family_size_survival(df):
    """
    Survival by family size
    From notebook section 4.2.2.6
    """
    if df is None or 'Family_Size' not in df.columns:
        return None
    
    family_survival = df.groupby('Family_Size')['Survived'].mean() * 100
    
    fig = go.Figure(data=[go.Scatter(
        x=family_survival.index,
        y=family_survival.values,
        mode='lines+markers',
        marker=dict(size=8, color=COLOR_PALETTES['primary']),
        line=dict(width=3)
    )])
    
    fig.update_layout(
        title='Survival Rate by Family Size',
        xaxis_title='Family Size',
        yaxis_title='Survival Rate (%)',
        height=400
    )
    
    return fig

# ----------------17. Alone vs Family (from notebook section 4.2.2.6)
def create_alone_vs_family_analysis(df):
    """
    Comparison between solo travelers and families
    From notebook section 4.2.2.6
    """
    if df is None or 'Is_Alone' not in df.columns:
        return None
    
    alone_survival = df.groupby('Is_Alone')['Survived'].mean() * 100
    labels = ['With Family', 'Alone']
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=alone_survival.values,
        marker_color=[COLOR_PALETTES['success'], COLOR_PALETTES['warning']],
        text=[f"{val:.1f}%" for val in alone_survival.values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Survival: Alone vs With Family',
        xaxis_title='Traveler Type',
        yaxis_title='Survival Rate (%)',
        height=400
    )
    
    return fig

# ----------------18. Detailed Family Composition (from notebook section 4.2.2.6)
def create_family_composition_analysis(df):
    """
    Detailed family composition analysis
    From notebook section 4.2.2.6
    """
    if df is None:
        return None
    
    # Analyze SibSp and Parch separately
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Survival by Siblings/Spouses', 'Survival by Parents/Children')
    )
    
    # SibSp analysis
    sibsp_survival = df.groupby('SibSp')['Survived'].mean() * 100
    fig.add_trace(
        go.Bar(x=sibsp_survival.index, y=sibsp_survival.values, 
               name='SibSp', marker_color=COLOR_PALETTES['primary']),
        row=1, col=1
    )
    
    # Parch analysis
    parch_survival = df.groupby('Parch')['Survived'].mean() * 100
    fig.add_trace(
        go.Bar(x=parch_survival.index, y=parch_survival.values, 
               name='Parch', marker_color=COLOR_PALETTES['secondary']),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

# ----------------19. Multivariate Analysis (combination of all factors)
def create_multivariate_survival_analysis(df):
    """
    Multivariate analysis of all factors
    """
    if df is None:
        return None
    
    # Heatmap survival by class and gender
    survival_matrix = df.groupby(['Pclass', 'Sex'])['Survived'].mean() * 100
    survival_pivot = survival_matrix.unstack()
    
    class_labels = [VALUE_MAPPINGS['Pclass'][pclass] for pclass in survival_pivot.index]
    gender_labels = [VALUE_MAPPINGS['Sex'][sex] for sex in survival_pivot.columns]
    
    fig = go.Figure(data=go.Heatmap(
        z=survival_pivot.values,
        x=gender_labels,
        y=class_labels,
        colorscale='RdYlGn',
        text=survival_pivot.round(1).values,
        texttemplate="%{text}%",
        textfont={"size": 12},
        colorbar=dict(title="Survival Rate (%)")
    ))
    
    fig.update_layout(
        title="Survival Rate by Class and Gender",
        xaxis_title="Gender",
        yaxis_title="Class",
        height=400
    )
    
    return fig

# ----------------20. Survival Factors Ranking
def calculate_survival_factors_ranking(df):
    """
    Calculate ranking of survival influence factors
    """
    if df is None:
        return None
    
    factors_impact = []
    
    # Calculate survival range for each factor
    
    # Gender
    gender_range = df.groupby('Sex')['Survived'].mean().max() - df.groupby('Sex')['Survived'].mean().min()
    factors_impact.append({'Factor': 'Gender', 'Impact_Range': gender_range * 100, 'Importance': 'Very High'})
    
    # Class
    class_range = df.groupby('Pclass')['Survived'].mean().max() - df.groupby('Pclass')['Survived'].mean().min()
    factors_impact.append({'Factor': 'Class', 'Impact_Range': class_range * 100, 'Importance': 'High'})
    
    # Age (if available)
    if 'Age_Group' in df.columns:
        age_range = df.groupby('Age_Group')['Survived'].mean().max() - df.groupby('Age_Group')['Survived'].mean().min()
        factors_impact.append({'Factor': 'Age Group', 'Impact_Range': age_range * 100, 'Importance': 'Medium'})
    
    # Family
    if 'Family_Size' in df.columns:
        family_range = df.groupby('Family_Size')['Survived'].mean().max() - df.groupby('Family_Size')['Survived'].mean().min()
        factors_impact.append({'Factor': 'Family Size', 'Impact_Range': family_range * 100, 'Importance': 'Medium'})
    
    # Fare
    if 'Fare_Category' in df.columns:
        fare_range = df.groupby('Fare_Category')['Survived'].mean().max() - df.groupby('Fare_Category')['Survived'].mean().min()
        factors_impact.append({'Factor': 'Fare Category', 'Impact_Range': fare_range * 100, 'Importance': 'Medium'})
    
    factors_df = pd.DataFrame(factors_impact)
    factors_df = factors_df.sort_values('Impact_Range', ascending=False)
    factors_df['Impact_Range'] = factors_df['Impact_Range'].round(1)
    
    return factors_df