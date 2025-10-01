from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px


def generate_chart():
    """Generate top 5 most hit episodes strip chart with PLR loudness values and save as HTML."""
    # Find the script directory and CSV file path
    script_dir = Path(__file__).parent
    csv_path = script_dir / "master_data.csv"

    # Load master_data.csv
    df = pd.read_csv(csv_path, on_bad_lines='skip')

    # Count gong hits per episode to find top 5
    episode_counts = df.groupby('Episode').size().reset_index(name='gong_hits')
    top_5_episodes = episode_counts.nlargest(5, 'gong_hits')['Episode'].tolist()

    # Filter original data to only include gong hits from top 5 episodes
    top_episodes_data = df[df['Episode'].isin(top_5_episodes)].copy()

    # Sort episodes by total gong hits for consistent ordering (reverse for 5th on left, 1st on right)
    episode_order = episode_counts.nlargest(5, 'gong_hits')['Episode'].tolist()
    episode_order.reverse()  # Reverse so 5th place is on the left, 1st place on the right

    # Create episode labels with day of week on first line, date on second line
    def format_episode_name(episode_name):
        # Split "Friday, June 20th" into "Friday,<br>June 20th"
        parts = episode_name.split(', ', 1)
        if len(parts) > 1:
            return f"{parts[0]},<br>{parts[1]}"
        return episode_name

    episode_labels = [format_episode_name(ep) for ep in episode_order]

    # Add jitter to x-coordinates for better dot separation
    np.random.seed(42)  # For reproducible jitter
    top_episodes_data_jittered = top_episodes_data.copy()

    # Add small random jitter to episode positions
    episode_positions = {ep: i for i, ep in enumerate(episode_order)}
    top_episodes_data_jittered['episode_numeric'] = top_episodes_data_jittered['Episode'].map(episode_positions)
    top_episodes_data_jittered['episode_jittered'] = (
        top_episodes_data_jittered['episode_numeric'] +
        np.random.uniform(-0.275, 0.275, len(top_episodes_data_jittered))  # Increased from ±0.15 to ±0.225 (50% more)
    )

    # Create strip chart
    fig = px.strip(
        top_episodes_data,
        x="Episode",
        y="plr_norm",
        color="Episode",
        hover_name="Gong Master",
        custom_data=["timestamped_link"],  # Add URL data for click functionality
        category_orders={"Episode": episode_order},
        stripmode="overlay"  # Back to overlay with jitter
    )

    # Style traces and add median lines
    for i, episode in enumerate(episode_order):
        if i < len(fig.data):
            trace = fig.data[i]
            episode_data = top_episodes_data[top_episodes_data["Episode"] == episode]

            if len(episode_data) > 0:
                median = episode_data["plr_norm"].median()

                # Update trace styling with smaller size for less overlap
                trace.update(
                    marker={
                        "color": "black",
                        "size": 8,  # Slightly smaller for less visual overlap
                        "opacity": 0.7,
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
                    x0=i-0.3, x1=i+0.3,
                    y0=median, y1=median,
                    line={"color": "black", "width": 2}
                )

                fig.add_annotation(
                    x=i + 0.35, y=median,
                    text=f"{median:.2f}",
                    showarrow=False,
                    font={"family": "gill sans", "size": 12, "color": "black"},
                    xanchor="left"
                )

    # Complete layout configuration using dimensions from sitting stripchart
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
            "tickfont": {"size": 12},
            "tickangle": 0,  # Keep episode names horizontal
            "ticktext": episode_labels,  # Use clean labels without day of week
            "tickvals": list(range(len(episode_order)))  # Position labels correctly
        },
        yaxis={
            "range": [.175, 1.025],
            "tickmode": 'array',
            "tickvals": [0.2, 0.4, 0.6, 0.8, 1],
            "ticktext": ['.2', '.4', '.6', '.8', '1'],
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
    output_path = output_dir / "10_gong_hits_strip.html"
    with open(output_path, "w") as f:
        f.write(html_with_clicks)

    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    generate_chart()
