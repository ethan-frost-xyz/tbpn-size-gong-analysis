import plotly.express as px
import pandas as pd
import numpy as np
import os
from pathlib import Path


def generate_chart():
    """Generate daily gong hits boxplot and save as PNG."""
    
    # Find the script directory and CSV file path
    script_dir = Path(__file__).parent
    csv_path = script_dir / "master_data.csv"
    
    # Load master_data.csv
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    # Convert the date column to datetime for day extraction
    df['date_parsed'] = pd.to_datetime(df['date_formated_other'], format='%m/%d/%y')
    
    # Extract day of the week
    df['day_of_week'] = df['date_parsed'].dt.day_name()
    
    # Count gong hits per episode (group by Episode and count rows)
    episode_counts = df.groupby(['Episode', 'day_of_week']).size().reset_index(name='gong_hits')
    
    # Filter for weekdays only (Monday through Friday)
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    episode_counts = episode_counts[episode_counts['day_of_week'].isin(weekdays)]
    
    # Calculate total episodes per day for labels
    day_totals = episode_counts.groupby('day_of_week').size()
    
    # Create custom labels with totals
    custom_labels = []
    for day in weekdays:
        total = day_totals.get(day, 0)
        custom_labels.append(f"{day}<br>N={total}")
    
    # Create minimalist boxplot with color separation
    fig = px.box(
        episode_counts,
        x="day_of_week",
        y="gong_hits",
        points=False,  # Remove scatter plot points
        color="day_of_week",  # This creates separate traces for each day
        category_orders={"day_of_week": weekdays},
    )
    
    # Minimalist styling with center alignment
    fig.update_layout(
        template="simple_white",
        font=dict(family="monotype bembo", size=14, color="black"),
        xaxis_title="",
        yaxis_title="Gong Hits in Episode",
        showlegend=False,
        width=647.2,
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode=False,  # Completely disable hover
        dragmode=False,  # Disable drag interactions
    )
    
    # Update axis properties separately to ensure font consistency
    fig.update_xaxes(
        showgrid=False,
        showline=False,  # Remove axis line for minimalism
        zeroline=False,
        tickfont=dict(size=14, family="monotype bembo", color="black"),
        ticktext=custom_labels,
        tickvals=list(range(len(weekdays)))
    )
    
    fig.update_yaxes(
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
        tickfont=dict(size=14, family="monotype bembo", color="black"),
        ticktext=custom_labels,
        tickvals=list(range(len(weekdays)))
    )
    
    # Minimalist box styling with white fill for all days
    for trace in fig.data:
        trace.update(
            fillcolor="white",
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
    
    # Add value labels for each boxplot (both quartiles)
    for i, day in enumerate(weekdays):
        data_subset = episode_counts[episode_counts["day_of_week"] == day]["gong_hits"]
        
        if len(data_subset) > 0:  # Only add labels if we have data
            # Calculate statistics
            q1 = data_subset.quantile(0.25)
            median = data_subset.median()
            q3 = data_subset.quantile(0.75)
            
            # Add annotations for key values
            x_pos = i  # Position based on day index
            offset = 0.275  # Horizontal offset from center
            
            # Median label
            fig.add_annotation(
                x=x_pos + offset, y=median,
                text=f"{median:.1f}",
                showarrow=False,
                font=dict(family="gill sans", size=12, color="black"),
                xanchor="left"
            )
            
            # Q1 label
            fig.add_annotation(
                x=x_pos + offset, y=q1,
                text=f"{q1:.1f}",
                showarrow=False,
                font=dict(family="gill sans", size=12, color="black"),
                xanchor="left"
            )
            
            # Q3 label
            fig.add_annotation(
                x=x_pos + offset, y=q3,
                text=f"{q3:.1f}",
                showarrow=False,
                font=dict(family="gill sans", size=12, color="black"),
                xanchor="left"
            )
    
    # Create output directory if it doesn't exist
    output_dir = script_dir / "charts_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save as PNG with high resolution
    output_path = output_dir / "9_daily_gong_hits.png"
    fig.write_image(output_path, 
                    width=647.2, 
                    height=400, 
                    scale=2)  # Higher scale for better quality
    
    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    generate_chart()
