import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path


def generate_chart():
    """Generate John (sitting vs standing) vs Jordi funding stage bar chart and save as PNG."""
    
    # Find the script directory and CSV file path
    script_dir = Path(__file__).parent
    csv_path = script_dir / "funding_only.csv"
    
    # Load funding_only.csv
    df = pd.read_csv(csv_path)
    
    # Create separate DataFrames for John and Jordi
    john_df = df[df['Gong Master'] == 'John'].copy()
    john_df['John_Status'] = john_df['john_sitting'].apply(lambda x: 'Sitting' if x == 1 else 'Standing')
    
    jordi_df = df[df['Gong Master'] == 'Jordi'].copy()
    
    # Count data for each category
    john_standing_counts = john_df[john_df['John_Status'] == 'Standing']['Funding Stage'].value_counts()
    john_sitting_counts = john_df[john_df['John_Status'] == 'Sitting']['Funding Stage'].value_counts()
    jordi_counts = jordi_df['Funding Stage'].value_counts()
    
    # Define stages in order
    stages = ["Seed", "Series A", "Series B", "Series C", "Series D or Later"]
    
    # Create figure
    fig = go.Figure()
    
    # Add John Sitting bars (base layer)
    fig.add_trace(go.Bar(
        name='John Sitting',
        x=stages,
        y=[john_sitting_counts.get(stage, 0) for stage in stages],
        marker_color='#1f78b4',
        offsetgroup=0,  # John's group
        base=0  # Start from bottom
    ))
    
    # Add John Standing bars (stacked on top)
    fig.add_trace(go.Bar(
        name='John Standing',
        x=stages,
        y=[john_standing_counts.get(stage, 0) for stage in stages],
        marker_color='#a6cee3',
        offsetgroup=0,  # Same group as John Sitting
        base=[john_sitting_counts.get(stage, 0) for stage in stages]  # Stack on top of sitting
    ))
    
    # Add Jordi bars (separate group)
    fig.add_trace(go.Bar(
        name='Jordi',
        x=stages,
        y=[jordi_counts.get(stage, 0) for stage in stages],
        marker_color='#b2df8a',
        offsetgroup=1  # Separate group from John
    ))
    
    # Minimalist styling
    fig.update_layout(
        template="simple_white",
        font=dict(family="monotype bembo", size=14, color="black"),
        xaxis_title="",  # Remove x-axis title for minimalism
        yaxis_title="Gong Hits",
        showlegend=False,
        width=647.2,
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
            tickfont=dict(size=14),
            tickangle=-45  # Angle the stage labels
        ),
        yaxis=dict(
            range=[-0.5, 20],
            tickmode='array',
            tickvals=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            showgrid=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            zeroline=False,
            tickfont=dict(size=14)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    
    # Update bar colors and styling
    fig.update_traces(
        marker=dict(
            line=dict(width=1, color="black"),
            opacity=1
        ),
        hoverinfo="none"  # Disable hover
    )
    
    # Set specific colors for John standing, John sitting, and Jordi
    colors = {
        "John (Standing)": "#a6cee3",  # Light blue for standing
        "John (Sitting)": "#1f78b4",   # Dark blue for sitting
        "Jordi": "#b2df8a"             # Light green for Jordi
    }
    for trace in fig.data:
        if trace.name in colors:
            trace.marker.color = colors[trace.name]
    
    # Add Tufte-style direct labels with thin black lines for Series A
    # Use actual data coordinates instead of random positioning
    series_a_category = "Series A"
    
    # Get Series A values for positioning
    john_sitting_series_a = john_sitting_counts.get("Series A", 0)
    john_standing_series_a = john_standing_counts.get("Series A", 0)
    jordi_series_a = jordi_counts.get("Series A", 0)
    
    # John Sitting
    fig.add_annotation(
        x=series_a_category,  # Use actual category name
        y=john_sitting_series_a / 1.2, 
        xref="x",
        yref="y",
        text="John Sitting",
        showarrow=True,
        arrowhead=0,
        arrowcolor="black",
        arrowwidth=1,
        arrowsize=0.5,
        ax=50,  
        ay=-45,   
        xanchor="left",
        yanchor="middle",
        font=dict(family="gill sans", size=12, color="black"),
        bgcolor="white",
        bordercolor="rgba(0,0,0,0)",
        borderwidth=0
    )
    
    # John Standing
    fig.add_annotation(
        x=series_a_category,  # Use actual category name
        y=john_sitting_series_a + (john_standing_series_a / 2),  # Middle of standing bar
        xref="x",
        yref="y",
        text="John Standing",
        showarrow=True,
        arrowhead=0,
        arrowcolor="black",
        arrowwidth=1,
        arrowsize=0.5,
        ax=50,  # Position text to the right
        ay=-15,   # No vertical offset from bar
        xanchor="left",
        yanchor="middle",
        font=dict(family="gill sans", size=12, color="black"),
        bgcolor="white",
        bordercolor="rgba(0,0,0,0)",
        borderwidth=0
    )
    
    # Jordi
    fig.add_annotation(
        x=1.4,  # Jordi bar x-position (offsetgroup=1 creates offset)
        y=jordi_series_a/1.2,  # Middle of Jordi bar
        xref="x",
        yref="y",
        text="Jordi",
        showarrow=True,
        arrowhead=0,
        arrowcolor="black",
        arrowwidth=1,
        arrowsize=0.5,
        ax=30,  # Position text to the right (same side as other labels)
        ay=-20,   # No vertical offset from bar
        xanchor="left",
        yanchor="middle",
        font=dict(family="gill sans", size=12, color="black"),
        bgcolor="white",
        bordercolor="rgba(0,0,0,0)",
        borderwidth=0
    )
    
    # Create output directory if it doesn't exist
    output_dir = script_dir / "charts_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save as PNG with high resolution
    output_path = output_dir / "5_sitting_barchart.png"
    fig.write_image(output_path, 
                    width=647.2, 
                    height=400, 
                    scale=2)  # Higher scale for better quality
    
    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    generate_chart()
