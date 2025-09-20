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
        name='John (Sitting)',
        x=stages,
        y=[john_sitting_counts.get(stage, 0) for stage in stages],
        marker_color='#90EE90',
        offsetgroup=0,  # John's group
        base=0  # Start from bottom
    ))
    
    # Add John Standing bars (stacked on top)
    fig.add_trace(go.Bar(
        name='John (Standing)',
        x=stages,
        y=[john_standing_counts.get(stage, 0) for stage in stages],
        marker_color='#009966',
        offsetgroup=0,  # Same group as John Sitting
        base=[john_sitting_counts.get(stage, 0) for stage in stages]  # Stack on top of sitting
    ))
    
    # Add Jordi bars (separate group)
    fig.add_trace(go.Bar(
        name='Jordi',
        x=stages,
        y=[jordi_counts.get(stage, 0) for stage in stages],
        marker_color='#ffe020',
        offsetgroup=1  # Separate group from John
    ))
    
    # Minimalist styling
    fig.update_layout(
        template="simple_white",
        font=dict(family="Times New Roman", size=12, color="black"),
        xaxis_title="",  # Remove x-axis title for minimalism
        yaxis_title="Value",
        showlegend=True,
        width=647.2,
        height=400,
        margin=dict(l=40, r=20, t=20, b=60),  # Extra bottom margin for stage labels
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            zeroline=False,
            tickfont=dict(size=10),
            tickangle=-45  # Angle the stage labels
        ),
        yaxis=dict(
            range=[0, 20],
            tickmode='array',
            tickvals=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            showgrid=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            zeroline=False,
            tickfont=dict(size=12)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            x=0.65,
            y=0.9,
            bgcolor="rgba(255,255,255,0)",  # Transparent legend background
            bordercolor="rgba(0,0,0,0)",
            font=dict(family="Times New Roman", size=10, color="black")
        )
    )
    
    # Update bar colors and styling
    fig.update_traces(
        marker=dict(
            line=dict(width=0.5, color="black"),
            opacity=0.8
        ),
        hoverinfo="none"  # Disable hover
    )
    
    # Set specific colors for John standing, John sitting, and Jordi
    colors = {
        "John (Standing)": "#009966",  # Original John green
        "John (Sitting)": "#90EE90",   # Light green for sitting
        "Jordi": "#ffe020"             # Yellow for Jordi
    }
    for trace in fig.data:
        if trace.name in colors:
            trace.marker.color = colors[trace.name]
    
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
