import plotly.express as px
import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def generate_chart():
    """Generate funding vs loudness scatter plot with trendline and save as HTML."""
    
    # Find the script directory and CSV file path
    script_dir = Path(__file__).parent
    csv_path = script_dir / "funding_only.csv"
    
    # Load data and create scatter plot with trendline
    df = pd.read_csv(csv_path)
    fig = px.scatter(
        df,
        x="Amount Raised",
        y="plr_norm",
        hover_name="Company",
        custom_data=["timestamped_link"],  # Add URL data for click functionality
        trendline="ols",
        title=""
    )
    
    # Style trendline and markers in single updates
    fig.update_traces(
        selector=dict(mode="lines"),
        line=dict(color="black", width=2, dash="dot"),
        name="OLS Fit",
        hoverinfo="skip",
        hovertemplate=None
    )
    
    fig.update_traces(
        selector=dict(mode="markers"),
        marker=dict(color="black", size=5, opacity=.8, line=dict(width=0)),
        hovertemplate="<span style='font-weight: 600; font-family: gill sans;'>%{hovertext}</span><br>$%{x:.1f}M<br>PLR: %{y:.2f}<extra></extra>",
        hoverlabel=dict(
            font=dict(family="gill sans", color="black"),
            bgcolor="white",
            bordercolor="black"
        )
    )
    
    # Calculate R² for annotation
    X = sm.add_constant(df['Amount Raised'])
    r2 = sm.OLS(df['plr_norm'], X).fit().rsquared
    
    # Complete layout configuration
    fig.update_layout(
        template="simple_white",
        font=dict(family="monotype bembo", size=14, color="black"),
        xaxis_title="Funding ($M)",
        yaxis_title="Loudness",
        showlegend=False,
        width=647.2,
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        title_font=dict(size=18),
        dragmode=False,
        xaxis=dict(
            range=[0, 300],
            showgrid=False,
            showline=True,
            zeroline=False,
            tickfont=dict(size=14),
            fixedrange=True
        ),
        yaxis=dict(
            range=[.2, 1],
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
    
    # Add R² annotation
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.92,
        text=f"R² = {r2:.2f}",
        showarrow=False,
        font=dict(size=12, family="gill sans", color="black")
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
    
    # Create output directory if it doesn't exist
    output_dir = script_dir / "charts_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save the interactive HTML file
    output_path = output_dir / "1_the_results.html"
    with open(output_path, "w") as f:
        f.write(html_with_clicks)
    
    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    generate_chart()
