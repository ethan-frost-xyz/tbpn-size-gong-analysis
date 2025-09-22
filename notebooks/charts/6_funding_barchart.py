import plotly.express as px
import pandas as pd
from pathlib import Path


def generate_chart():
    """Generate John sitting percentage by funding threshold bar chart and save as PNG."""
    
    # Find the script directory and CSV file path
    script_dir = Path(__file__).parent
    csv_path = script_dir / "funding_only.csv"
    
    # Load data and calculate sitting percentages by funding threshold
    df = pd.read_csv(csv_path)
    john_df = df[df["Gong Master"] == "John"]
    
    # Calculate sitting percentages for funding thresholds
    thresholds = [0, 10, 20, 30, 40]
    threshold_labels = ["All Rounds", "Above $10M", "Above $20M", "Above $30M", "Above $40M"]
    
    sitting_percentages = [
        john_df[john_df["Amount Raised"] > threshold]["john_sitting"].mean() 
        if len(john_df[john_df["Amount Raised"] > threshold]) > 0 else 0
        for threshold in thresholds
    ]
    
    # Create bar chart
    fig = px.bar(x=threshold_labels, y=sitting_percentages, title="")
    
    # Minimalist styling
    fig.update_layout(
        template="simple_white",
        font=dict(family="monotype bembo", size=14, color="black"),
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        width=647.2,
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
            tickfont=dict(size=14),
            tickangle=-45
        ),
        yaxis=dict(
            range=[-0.01, 0.4],
            tickmode='array',
            tickvals=[0, 0.1, 0.2, 0.3, 0.4],
            ticktext=['0.0%', '10.0%', '20.0%', '30.0%', '40.0%'],
            showgrid=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            zeroline=False,
            tickfont=dict(size=14)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    # Bar styling
    fig.update_traces(
        marker=dict(
            color="#1f78b4",
            line=dict(width=1, color="black"),
            opacity=1
        ),
        hoverinfo="none"
    )
    
    # Add percentage labels (only for first and last bars)
    for i, percentage in enumerate(sitting_percentages):
        if i == 0 or i == len(sitting_percentages) - 1:  # Only first and last
            fig.add_annotation(
                x=i, y=percentage + 0.005,
                text=f"{percentage:.1%}",
                showarrow=False,
                font=dict(family="Gill Sans", size=12, color="black"),
                xanchor="center",
                yanchor="bottom"
            )
    
    # Create output directory if it doesn't exist
    output_dir = script_dir / "charts_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save as PNG with high resolution
    output_path = output_dir / "6_funding_barchart.png"
    fig.write_image(output_path, 
                    width=647.2, 
                    height=400, 
                    scale=2)  # Higher scale for better quality
    
    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    generate_chart()
