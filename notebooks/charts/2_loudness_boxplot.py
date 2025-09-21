import plotly.express as px
import pandas as pd
import numpy as np
import os
from pathlib import Path


def generate_chart():
    """Generate John vs Jordi loudness boxplot and save as PNG."""
    
    # Find the script directory and CSV file path
    script_dir = Path(__file__).parent
    csv_path = script_dir / "funding_only.csv"
    
    # Load funding_only.csv
    df = pd.read_csv(csv_path)
    
    # Filter for John and Jordi only
    df_filtered = df[df["Gong Master"].isin(["John", "Jordi"])]
    
    # Box styling parameters
    boxgap = 0.6  # Controls spacing between boxes
    
    # Create minimalist boxplot with color separation
    fig = px.box(
        df_filtered,
        x="Gong Master",
        y="plr_norm",
        points=False,  # Remove scatter plot points
        color="Gong Master",  # This creates separate traces for each person
        category_orders={"Gong Master": ["John", "Jordi"]},
    )
    
    # Minimalist styling with center alignment
    fig.update_layout(
        template="simple_white",
        font=dict(family="monotype bembo", size=14, color="black"),
        xaxis_title="",
        yaxis_title="Loudness",
        showlegend=False,
        width=700,
        height=500,
        margin=dict(l=10, r=10, t=10, b=40),  # Account for axis labels
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode=False,  # Completely disable hover
        dragmode=False,  # Disable drag interactions
        boxgap=boxgap,
    )
    
    # Update axis properties separately to ensure font consistency
    fig.update_xaxes(
        showgrid=False,
        showline=False,  # Remove axis line for minimalism
        zeroline=False,
        tickfont=dict(size=14, family="monotype bembo", color="black")
    )
    
    fig.update_yaxes(
        range=[.2, 1],
        tickmode='array',
        tickvals=[.2, .4, .6, .8, 1],
        showgrid=False,
        showline=True,
        linecolor="white",
        linewidth=1,
        zeroline=False,
        tickfont=dict(size=14, family="monotype bembo", color="black")
    )
    
    # Set global font first
    fig.update_layout(
        font=dict(size=14, family="monotype bembo", color="black")
    )
    
    # Then override x-axis category labels to be larger
    fig.update_xaxes(
        tickfont=dict(size=14, family="monotype bembo", color="black")
    )
    
    # Minimalist box styling with different colors for each person
    colors = {"John": "white", "Jordi": "white"}
    for trace in fig.data:
        # Extract the gong master name from the trace name
        gong_master = trace.name
        trace.update(
            fillcolor=colors.get(gong_master, "white"),
            line=dict(color="black", width=.75),
            marker=dict(
                color="black",
                size=4,
                opacity=0.6
            ),
            boxpoints=False,  # Hide all points
            hoverinfo="skip",  # Completely skip hover
            hovertemplate=None  # Remove hover template
        )
    
    # Add value labels for each boxplot
    for i, gong_master in enumerate(["John", "Jordi"]):
        data_subset = df_filtered[df_filtered["Gong Master"] == gong_master]["plr_norm"]
        
        # Calculate statistics
        q1 = data_subset.quantile(0.25)
        median = data_subset.median()
        q3 = data_subset.quantile(0.75)
        min_val = data_subset.min()
        max_val = data_subset.max()
        
        # Add annotations for key values
        x_pos = i  # 0 for John, 1 for Jordi
        # Calculate offset based on box width: higher boxgap = narrower boxes = closer offset
        box_width = (1 - boxgap) / 2  # Approximate box width
        offset = box_width / 2 + 0.05  # Half box width plus small buffer
        
        # Median label
        fig.add_annotation(
            x=x_pos + offset, y=median,
            text=f"{median:.2f}",
            showarrow=False,
            font=dict(family="gill sans", size=12, color="black"),
            xanchor="left"
        )
        
        # Q1 label
        fig.add_annotation(
            x=x_pos + offset, y=q1,
            text=f"{q1:.2f}",
            showarrow=False,
            font=dict(family="gill sans", size=12, color="black"),
            xanchor="left"
        )
    
    # Create output directory if it doesn't exist
    output_dir = script_dir / "charts_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save as PNG with high resolution
    output_path = output_dir / "2_loudness_boxplot.png"
    fig.write_image(output_path, 
                    width=700, 
                    height=500, 
                    scale=2)  # Higher scale for better quality
    
    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    generate_chart()
