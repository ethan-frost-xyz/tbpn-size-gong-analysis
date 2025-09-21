import plotly.express as px
import pandas as pd
from pathlib import Path


def generate_chart():
    """Generate John sitting vs standing loudness strip chart and save as PNG."""
    
    # Find the script directory and CSV file path
    script_dir = Path(__file__).parent
    csv_path = script_dir / "funding_only.csv"
    
    # Load data and prepare John's position data
    df = pd.read_csv(csv_path)
    john_df = df[df["Gong Master"] == "John"].copy()
    john_df["Position"] = john_df["john_sitting"].apply(lambda x: "Sitting" if x == 1 else "Standing")
    
    # Create swarm chart
    fig = px.strip(
        john_df,
        x="Position",
        y="plr_norm",
        color="Position",
        hover_name="Company",
        category_orders={"Position": ["Standing", "Sitting"]},
        stripmode="overlay"
    )
    
    # Define colors and prepare position data once
    colors = {"Standing": "#a6cee3", "Sitting": "#1f78b4"}
    position_data = {pos: john_df[john_df["Position"] == pos] for pos in ["Standing", "Sitting"]}
    
    # Style traces and add median lines in single loop
    for i, (position, trace) in enumerate(zip(["Standing", "Sitting"], fig.data)):
        data_subset = position_data[position]
        median = data_subset["plr_norm"].median()
        
        # Update trace styling
        trace.update(
            marker=dict(
                color=colors[position],
                size=10,
                opacity=1,
                line=dict(width=0)
            ),
            hovertemplate="%{hovertext}<br>Funding: $%{x:.0f}M<br>PLR: %{y:.2f}<extra></extra>",
            hoverlabel=dict(
                font=dict(family="monotype bembo", color="black"),
                bgcolor="white",
                bordercolor="black"
            )
        )
        
        # Add median line and label
        fig.add_shape(
            type="line",
            x0=i-0.2, x1=i+0.2,
            y0=median, y1=median,
            line=dict(color="black", width=2)
        )
        
        fig.add_annotation(
            x=i + 0.225, y=median,
            text=f"{median:.2f}",
            showarrow=False,
            font=dict(family="gill sans", size=12, color="black"),
            xanchor="left"
        )
    
    # Complete layout configuration
    fig.update_layout(
        template="simple_white",
        font=dict(family="monotype bembo", size=14, color="black"),
        xaxis_title="",
        yaxis_title="Loudness (PLR)",
        showlegend=False,
        width=600,
        height=600,
        margin=dict(l=80, r=80, t=40, b=60),
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            range=[.2, 1],
            tickmode='array',
            tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
            showgrid=False,
            showline=True,
            linecolor="white",
            linewidth=1,
            zeroline=False,
            tickfont=dict(size=14)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    # Create output directory if it doesn't exist
    output_dir = script_dir / "charts_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save as PNG with high resolution
    output_path = output_dir / "7_sitting_stripchart.png"
    fig.write_image(output_path, 
                    width=550, 
                    height=500, 
                    scale=2)  # Higher scale for better quality
    
    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    generate_chart()
