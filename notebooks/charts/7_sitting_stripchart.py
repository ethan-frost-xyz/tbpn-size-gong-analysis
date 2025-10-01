from pathlib import Path

import pandas as pd
import plotly.express as px


def generate_chart():
    """Generate John sitting vs standing loudness strip chart and save as HTML."""
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
        custom_data=["timestamped_link"],  # Add URL data for click functionality
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
            marker={
                "color": colors[position],
                "size": 10,
                "opacity": 1,
                "line": {"width": 0}
            },
            hovertemplate="<span style='font-weight: 600; font-family: gill sans;'>%{hovertext}</span><br>PLR: %{y:.2f}<extra></extra>",
            hoverlabel={
                "font": {"family": "gill sans", "color": "black"},
                "bgcolor": "white",
                "bordercolor": "black"
            }
        )

        # Add median line and label
        fig.add_shape(
            type="line",
            x0=i-0.2, x1=i+0.2,
            y0=median, y1=median,
            line={"color": "black", "width": 2}
        )

        fig.add_annotation(
            x=i + 0.225, y=median,
            text=f"{median:.2f}",
            showarrow=False,
            font={"family": "gill sans", "size": 12, "color": "black"},
            xanchor="left"
        )

    # Complete layout configuration
    fig.update_layout(
        template="simple_white",
        font={"family": "monotype bembo", "size": 14, "color": "black"},
        xaxis_title="",
        yaxis_title="Loudness",
        showlegend=False,
        width=647.2,
        height=400,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "tickfont": {"size": 14}
        },
        yaxis={
            "range": [.2, 1],
            "tickmode": 'array',
            "tickvals": [0.2, 0.4, 0.6, 0.8, 1.0],
            "showgrid": False,
            "showline": True,
            "linecolor": "white",
            "linewidth": 1,
            "zeroline": False,
            "tickfont": {"size": 14}
        },
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    # Create output directory if it doesn't exist
    output_dir = script_dir / "charts_output"
    output_dir.mkdir(exist_ok=True)

    # Show chart with click functionality enabled
    config = {
        "displaylogo": False,
        "displayModeBar": False
    }

    # Write HTML with embedded JavaScript for click handling
    html_string = fig.to_html(config=config, include_plotlyjs=True)

    # Add JavaScript for click handling with debouncing
    click_js = '''
<script>
document.addEventListener('DOMContentLoaded', function() {
    var clickTimeout = null;

    setTimeout(function() {
        var plotDiv = document.querySelector('.js-plotly-plot');
        if (plotDiv) {
            plotDiv.on('plotly_click', function(data) {
                // Check if a click is already being processed
                if (clickTimeout) {
                    return;
                }

                if (data.points && data.points[0] && data.points[0].customdata) {
                    var url = data.points[0].customdata[0];
                    if (url) {
                        // Set debounce timer to prevent rapid successive clicks
                        clickTimeout = setTimeout(function() {
                            clickTimeout = null;
                        }, 300);

                        window.open(url, '_blank');
                    }
                }
            });
        }
    }, 100);
});
</script>
'''

    # Insert JavaScript before closing body tag
    html_with_clicks = html_string.replace('</body>', click_js + '</body>')

    # Save the interactive HTML file
    output_path = output_dir / "7_sitting_stripchart.html"
    with open(output_path, "w") as f:
        f.write(html_with_clicks)

    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    generate_chart()
