"""
src/components/advanced_charts.py
Specialized charts for advanced analysis and feature engineering
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.config import COLOR_PALETTES, COLUMN_LABELS, VALUE_MAPPINGS

# ----------------1. Advanced Correlation Matrix (from notebook section 4.1.2 extended)
def create_correlation_matrix(df, method='pearson'):
    """
    Create correlation matrix with specified method
    Extends notebook section 4.1.2
    """
    if df is None:
        return None
    
    # Select only numerical variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'PassengerId']
    
    if len(numeric_cols) < 2:
        return None
    
    # Calculate correlation
    corr_matrix = df[numeric_cols].corr(method=method)
    
    # Mask upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_masked = corr_matrix.mask(mask)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_masked.values,
        x=[COLUMN_LABELS.get(col, col) for col in corr_matrix.columns],
        y=[COLUMN_LABELS.get(col, col) for col in corr_matrix.index],
        colorscale='RdBu',
        zmid=0,
        # Remove numbers from chart for better readability
        text=None,
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
        colorbar=dict(title=f"{method.title()} Correlation")
    ))
    
    fig.update_layout(
        title=f"{method.title()} Correlation Matrix",
        height=500,
        # Improve readability
        xaxis_tickangle=-45,
        margin=dict(l=100, r=50, t=80, b=100)
    )
    
    return fig

# ----------------2. Target Correlations
def create_target_correlation_chart(correlations_df):
    """
    Chart of correlations with target variable
    """
    if correlations_df is None:
        return None
    
    # Take top 10 correlations (absolute)
    top_corr = correlations_df.head(10)
    
    fig = go.Figure(data=[go.Bar(
        y=top_corr.index,
        x=top_corr.values,
        orientation='h',
        marker_color=COLOR_PALETTES['primary'],
        text=[f"{val:.3f}" for val in top_corr.values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Top Correlations with Survival",
        xaxis_title="Correlation",
        yaxis_title="Variables",
        height=400
    )
    
    return fig

# ----------------3. Correlations by Category
def create_correlation_by_category(df, category_col):
    """
    Separate correlation matrix by categories
    """
    if df is None or category_col not in df.columns:
        return None
    
    categories = df[category_col].unique()
    
    # Create subplot for each category
    fig = make_subplots(
        rows=1, cols=len(categories),
        subplot_titles=[VALUE_MAPPINGS.get(category_col, {}).get(cat, str(cat)) for cat in categories]
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['PassengerId']]
    
    for i, category in enumerate(categories):
        subset = df[df[category_col] == category]
        if len(subset) > 5:  # Only if we have enough data
            corr_matrix = subset[numeric_cols].corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=numeric_cols,
                    y=numeric_cols,
                    colorscale='RdBu',
                    zmid=0,
                    showscale=(i == len(categories)-1)  # Only last scale
                ),
                row=1, col=i+1
            )
    
    fig.update_layout(
        title=f"Correlations by {COLUMN_LABELS.get(category_col, category_col)}",
        height=400
    )
    
    return fig

# ----------------4. Title Analysis (from name feature engineering)
def create_title_survival_analysis(df):
    """
    Survival analysis by title extracted from name
    """
    if df is None or 'Title' not in df.columns:
        return None
    
    title_survival = df.groupby('Title')['Survived'].mean() * 100
    title_counts = df['Title'].value_counts()
    
    # Filter titles with at least 5 occurrences
    common_titles = title_counts[title_counts >= 5].index
    title_survival_filtered = title_survival[common_titles]
    
    fig = go.Figure(data=[go.Bar(
        x=title_survival_filtered.index,
        y=title_survival_filtered.values,
        marker_color=COLOR_PALETTES['primary'],
        text=[f"{val:.1f}%" for val in title_survival_filtered.values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Survival by Title",
        xaxis_title="Title",
        yaxis_title="Survival Rate (%)",
        height=400
    )
    
    return fig

# ----------------5. Deck Analysis (from cabin feature engineering)
def create_deck_survival_analysis(df):
    """
    Survival analysis by deck extracted from cabin
    """
    if df is None or 'Deck' not in df.columns:
        return None
    
    deck_survival = df.groupby('Deck')['Survived'].mean() * 100
    deck_counts = df['Deck'].value_counts()
    
    # Filter decks with at least 3 occurrences
    common_decks = deck_counts[deck_counts >= 3].index
    deck_survival_filtered = deck_survival[common_decks]
    
    fig = go.Figure(data=[go.Bar(
        x=deck_survival_filtered.index,
        y=deck_survival_filtered.values,
        marker_color=COLOR_PALETTES['secondary'],
        text=[f"{val:.1f}%" for val in deck_survival_filtered.values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Survival by Deck",
        xaxis_title="Deck",
        yaxis_title="Survival Rate (%)",
        height=400
    )
    
    return fig

# ----------------6. Feature Importance Chart
def create_feature_importance_chart(importance_df):
    """
    Feature importance chart
    """
    if importance_df is None:
        return None
    
    # Top 15 features
    top_features = importance_df.head(15)
    
    fig = go.Figure(data=[go.Bar(
        y=top_features['Feature'],
        x=top_features['Importance'],
        orientation='h',
        marker_color=COLOR_PALETTES['success'],
        text=[f"{val:.3f}" for val in top_features['Importance']],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Feature Importance (Proxy)",
        xaxis_title="Importance",
        yaxis_title="Features",
        height=500
    )
    
    return fig

# ----------------7. Outliers Scatter Plot (from extended notebook)
def create_outliers_scatter_plot(df, var1, var2):
    """
    Scatter plot with highlighted outliers
    Extends outlier analysis from notebook section 4.1.1
    """
    if df is None or var1 not in df.columns or var2 not in df.columns:
        return None
    
    from src.utils.data_processor import detect_outliers_iqr
    
    # Detect outliers for both variables
    outliers1, _, _ = detect_outliers_iqr(df[var1].dropna())
    outliers2, _, _ = detect_outliers_iqr(df[var2].dropna())
    
    # Combine outlier indices
    outlier_indices = set(outliers1.index) | set(outliers2.index)
    
    # Create outlier indicator
    df_plot = df.copy()
    df_plot['Is_Outlier'] = df_plot.index.isin(outlier_indices)
    
    fig = px.scatter(
        df_plot,
        x=var1,
        y=var2,
        color='Is_Outlier',
        color_discrete_map={False: COLOR_PALETTES['primary'], True: COLOR_PALETTES['danger']},
        title=f"Outliers: {COLUMN_LABELS.get(var1, var1)} vs {COLUMN_LABELS.get(var2, var2)}"
    )
    
    fig.update_layout(height=400)
    return fig

# ----------------8. Outliers Comparison Boxplot
def create_outliers_comparison_boxplot(df, variables):
    """
    Boxplot comparison of multiple variables for outliers
    """
    if df is None or not variables:
        return None
    
    fig = go.Figure()
    
    # Use direct colors
    default_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98FB98']
    
    for i, var in enumerate(variables):
        if var in df.columns:
            fig.add_trace(go.Box(
                y=df[var],
                name=COLUMN_LABELS.get(var, var),
                marker_color=default_colors[i % len(default_colors)]
            ))
    
    fig.update_layout(
        title="Distribution Comparison for Outliers",
        yaxis_title="Values",
        height=400
    )
    
    return fig

# ----------------9. Normality Test Plots
def create_normality_test_plots(df, variable):
    """
    Plots for normality tests
    """
    if df is None or variable not in df.columns:
        return None
    
    data = df[variable].dropna()
    
    # Create subplot with histogram and Q-Q plot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribution', 'Normal Q-Q Plot')
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=data, nbinsx=20, name="Distribution", opacity=0.7),
        row=1, col=1
    )
    
    # Approximated Q-Q plot
    sorted_data = np.sort(data)
    n = len(sorted_data)
    theoretical_quantiles = np.random.normal(data.mean(), data.std(), n)
    theoretical_quantiles.sort()
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles, 
            y=sorted_data, 
            mode='markers',
            name="Q-Q Plot",
            marker_color=COLOR_PALETTES['warning']
        ),
        row=1, col=2
    )
    
    # Reference line for Q-Q plot
    min_val = min(min(theoretical_quantiles), min(sorted_data))
    max_val = max(max(theoretical_quantiles), max(sorted_data))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name="Normal Line",
            line=dict(dash='dash', color='red')
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"Normality Test - {COLUMN_LABELS.get(variable, variable)}",
        height=400,
        showlegend=False
    )
    
    return fig

# ----------------10. Distribution Comparison by Group
def create_distribution_comparison_by_group(df, numeric_var, group_var):
    """
    Compare distributions of numerical variable by groups
    """
    if df is None or numeric_var not in df.columns or group_var not in df.columns:
        return None
    
    fig = go.Figure()
    
    groups = df[group_var].unique()
    # Use direct colors instead of seaborn_palettes
    default_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98FB98', '#87CEEB', '#DDA0DD']
    colors = default_colors[:len(groups)]
    
    for i, group in enumerate(groups):
        group_data = df[df[group_var] == group][numeric_var].dropna()
        group_label = VALUE_MAPPINGS.get(group_var, {}).get(group, str(group))
        
        fig.add_trace(go.Histogram(
            x=group_data,
            name=group_label,
            opacity=0.7,
            nbinsx=15,
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title=f"{COLUMN_LABELS.get(numeric_var, numeric_var)} Distribution by {COLUMN_LABELS.get(group_var, group_var)}",
        xaxis_title=COLUMN_LABELS.get(numeric_var, numeric_var),
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )
    
    return fig

# ----------------11. Segmentation Charts
def create_segments_survival_chart(df):
    """
    Survival chart by segments
    """
    if df is None or 'Segment' not in df.columns:
        return None
    
    segment_survival = df.groupby('Segment')['Survived'].mean() * 100
    
    fig = go.Figure(data=[go.Bar(
        x=segment_survival.index,
        y=segment_survival.values,
        marker_color=COLOR_PALETTES['age_groups'],
        text=[f"{val:.1f}%" for val in segment_survival.values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Survival Rate by Segment",
        xaxis_title="Segment",
        yaxis_title="Survival Rate (%)",
        height=400
    )
    
    return fig

# ----------------12. Segments Distribution
def create_segments_distribution_chart(df):
    """
    Distribution of segments in dataset
    """
    if df is None or 'Segment' not in df.columns:
        return None
    
    segment_counts = df['Segment'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=segment_counts.index,
        values=segment_counts.values,
        hole=.3
    )])
    
    fig.update_layout(
        title="Passenger Segments Distribution",
        height=400
    )
    
    return fig

# ----------------13. AFC Profiles Charts (Age-Fare-Class)
def create_profiles_chart(profile_survival):
    """
    Chart for Age-Fare-Class profiles
    """
    if profile_survival is None:
        return None
    
    # Bubble chart: size = count, y = survival rate
    fig = go.Figure(data=go.Scatter(
        x=profile_survival.index,
        y=profile_survival['Survival_Rate'] * 100,
        mode='markers',
        marker=dict(
            size=profile_survival['Count'],
            sizemode='diameter',
            sizeref=2.*max(profile_survival['Count'])/(40.**2),
            sizemin=4,
            color=profile_survival['Survival_Rate'] * 100,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Survival Rate (%)")
        ),
        text=[f"Count: {cnt}<br>Rate: {rate*100:.1f}%" 
              for cnt, rate in zip(profile_survival['Count'], profile_survival['Survival_Rate'])],
        hovertemplate='Profile: %{x}<br>%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Age-Fare-Class Profiles",
        xaxis_title="Profile",
        yaxis_title="Survival Rate (%)",
        height=400
    )
    
    return fig

# ----------------14. Survival Patterns Heatmap
def create_survival_patterns_heatmap(df, var1, var2):
    """
    Heatmap of survival patterns for 2 variables
    """
    if df is None or var1 not in df.columns or var2 not in df.columns:
        return None
    
    # Calculate survival rate by combinations
    survival_matrix = df.groupby([var1, var2])['Survived'].mean() * 100
    survival_pivot = survival_matrix.unstack(fill_value=0)
    
    # Map labels if available
    x_labels = [VALUE_MAPPINGS.get(var2, {}).get(col, str(col)) for col in survival_pivot.columns]
    y_labels = [VALUE_MAPPINGS.get(var1, {}).get(idx, str(idx)) for idx in survival_pivot.index]
    
    fig = go.Figure(data=go.Heatmap(
        z=survival_pivot.values,
        x=x_labels,
        y=y_labels,
        colorscale='RdYlGn',
        text=survival_pivot.round(1).values,
        texttemplate="%{text}%",
        textfont={"size": 10},
        colorbar=dict(title="Survival Rate (%)")
    ))
    
    fig.update_layout(
        title=f"Survival Patterns: {COLUMN_LABELS.get(var1, var1)} vs {COLUMN_LABELS.get(var2, var2)}",
        xaxis_title=COLUMN_LABELS.get(var2, var2),
        yaxis_title=COLUMN_LABELS.get(var1, var1),
        height=400
    )
    
    return fig

# ----------------15. Segments Radar Chart
def create_segments_radar_chart(df, segments_col='Segment'):
    """
    Radar chart to compare segment characteristics
    """
    if df is None or segments_col not in df.columns:
        return None
    
    # Variables for radar chart
    numeric_vars = ['Age', 'Fare', 'Family_Size', 'Survived']
    available_vars = [var for var in numeric_vars if var in df.columns]
    
    if len(available_vars) < 3:
        return None
    
    segments = df[segments_col].unique()
    
    fig = go.Figure()
    
    for segment in segments:
        segment_data = df[df[segments_col] == segment]
        
        # Calculate normalized means (0-1)
        values = []
        for var in available_vars:
            if var == 'Survived':
                val = segment_data[var].mean()  # Already 0-1
            else:
                val = (segment_data[var].mean() - df[var].min()) / (df[var].max() - df[var].min())
            values.append(val)
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=available_vars + [available_vars[0]],
            fill='toself',
            name=f'Segment {segment}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Segment Characteristics Comparison",
        height=500
    )
    
    return fig