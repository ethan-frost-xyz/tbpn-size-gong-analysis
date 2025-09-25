import plotly.express as px
import pandas as pd
from pathlib import Path


def generate_chart():
    """Generate greatest hits scatter plot with color coding for John vs Jordi."""
    
    # Find the script directory and CSV file path
    script_dir = Path(__file__).parent
    csv_path = script_dir / "greatest_hits.csv"
    
    # Load data - handle parsing errors by skipping bad lines
    df = pd.read_csv(csv_path, on_bad_lines='skip')

    # Add jitter to create horizontal strip effect
    import numpy as np
    np.random.seed(42)  # For consistent jitter
    df['jitter'] = np.random.normal(0, 0.025, len(df))
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x="jitter",  # Using jitter for x-axis to create horizontal strip
        y="plr_norm",
        hover_name="episode_title_clean",
        custom_data=["company_name", "youtube_timestamped_link", "gong_master"],  # Add company name, URL and gong master data
        title=""
    )
    
    # Define colors based on who hit the gong
    def get_marker_color(gong_master):
        if gong_master == 'john':
            return '#1f78b4'  # John sitting color (dark blue)
        elif gong_master == 'jordi':
            return '#b2df8a'  # Jordi color (light green)
        else:
            return '#1f78b4'  # Default to John sitting color for others
    
    # Apply colors to each point
    colors = [get_marker_color(master) for master in df['gong_master']]
    
    # Style markers
    fig.update_traces(
        marker=dict(color=colors, size=7.5, opacity=.8, line=dict(width=0)),  # 50% larger (5 * 1.5 = 7.5)
        hovertemplate="<span style='font-weight: 600; font-family: gill sans;'>%{customdata[0]}</span><br>%{hovertext}<br>PLR: %{y:.2f}<extra></extra>",
        hoverlabel=dict(
            font=dict(family="gill sans", color="black"),
            bgcolor="white",
            bordercolor="black"
        )
    )
    
    # Complete layout configuration
    fig.update_layout(
        template="simple_white",
        font=dict(family="monotype bembo", size=14, color="black"),
        xaxis_title="Personal Favorites",
        yaxis_title="Loudness",
        showlegend=False,
        width=647.2,
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        title_font=dict(size=18),
        dragmode=False,
        xaxis=dict(
            showgrid=False,
            showline=True,
            zeroline=False,
            tickfont=dict(size=14),
            fixedrange=True,
            showticklabels=False,  # Hide x-axis labels for horizontal strip
            ticks="",  # Hide tick marks/notches
            range=[-0.5, 0.5]  # Constrain x-axis range for horizontal strip
        ),
        yaxis=dict(
            range=[.175, 1.025],
            tickmode='array',
            tickvals=[0.2, 0.4, 0.6, 0.8, 1],
            ticktext=['.2', '.4', '.6', '.8', '1'],
            showgrid=False,
            showline=True,
            zeroline=False,
            tickfont=dict(size=14),
            fixedrange=True
        )
    )
    
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
                    var url = data.points[0].customdata[1];
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
    
    # Create output directory if it doesn't exist
    output_dir = script_dir / "charts_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save the interactive HTML file
    output_path = output_dir / "13_greatest_hits.html"
    with open(output_path, "w") as f:
        f.write(html_with_clicks)
    
    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    generate_chart()
