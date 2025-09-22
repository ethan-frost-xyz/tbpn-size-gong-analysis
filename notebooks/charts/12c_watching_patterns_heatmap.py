import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

def main():
    # Load data
    script_dir = Path(__file__).parent
    history_path = script_dir / "tbpn_viewing_timeline.csv"

    df = pd.read_csv(history_path)
    df['visit_timestamp'] = pd.to_datetime(df['visit_timestamp'])

    # Extract hour and day of week
    df['hour'] = df['visit_timestamp'].dt.hour
    df['day_of_week'] = df['visit_timestamp'].dt.day_name()

    # Order days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Group by day and hour - count TOTAL visits per time slot (not unique videos)
    heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='total_visits')
    heatmap_data = heatmap_data.pivot(index='day_of_week', columns='hour', values='total_visits').fillna(0)
    heatmap_data = heatmap_data.reindex(day_order)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Total Visits: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title="TBPN Video Visit Patterns by Day and Time",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        template='plotly_white',
        font=dict(family="monotype bembo", size=14, color="black"),
        width=1200,
        height=750,
        margin=dict(l=10, r=10, t=50, b=40)
    )

    fig.update_xaxes(
        tickvals=list(range(24)),
        ticktext=[f"{i}:00" for i in range(24)],
        showgrid=False,
        showline=True,
        linecolor="black",
        tickfont=dict(size=12, family="monotype bembo", color="black")
    )

    fig.update_yaxes(showgrid=False, showline=True, linecolor="black", tickfont=dict(size=12, family="monotype bembo", color="black"))

    # Save chart
    output_dir = script_dir / "charts_output"
    output_dir.mkdir(exist_ok=True)
    fig.write_image(output_dir / "12c_watching_patterns_heatmap.png", width=1200, height=750, scale=2)

if __name__ == "__main__":
    main()
