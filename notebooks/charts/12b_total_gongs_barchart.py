from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def main():
    # Load data
    script_dir = Path(__file__).parent
    history_path = script_dir / "tbpn_viewing_timeline.csv"

    df = pd.read_csv(history_path)
    df['visit_timestamp'] = pd.to_datetime(df['visit_timestamp'])

    # Extract day of week
    df['day_of_week'] = df['visit_timestamp'].dt.day_name()

    # Order days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Group by day of week - count TOTAL visits per day (not unique videos)
    day_data = df.groupby('day_of_week').size().reset_index(name='total_visits')
    day_data = day_data.set_index('day_of_week').reindex(day_order).reset_index()


    # Create bar chart
    fig = go.Figure(data=go.Bar(
        x=day_data['day_of_week'],
        y=day_data['total_visits'],
        marker_color='#1f78b4',
        marker_line={"width": 1, "color": "black"},
        hoverinfo="none"
    ))

    # Minimalist styling matching funding barchart
    fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="",
        template="simple_white",
        font={"family": "monotype bembo", "size": 14, "color": "black"},
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
            "range": [0, 700],
            "tickmode": 'array',
            "tickvals": list(range(0, 701, 100)),
            "ticktext": [f"{i}" for i in range(0, 701, 100)],
            "showgrid": False,
            "showline": True,
            "linecolor": "black",
            "linewidth": 1,
            "zeroline": False,
            "tickfont": {"size": 14}
        },
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    # Add value labels on bars
    for i, visits in enumerate(day_data['total_visits']):
        fig.add_annotation(
            x=i,
            y=visits + 0.5,  # Offset above bar
            text=f"{int(visits)}",  # Convert to int for clean display
            showarrow=False,
            font={"family": "Gill Sans", "size": 12, "color": "black"},
            xanchor="center",
            yanchor="bottom"
        )


    # Save chart
    output_dir = script_dir / "charts_output"
    output_dir.mkdir(exist_ok=True)
    fig.write_image(output_dir / "12b_total_gongs_barchart.png", width=647.2, height=400, scale=2)

if __name__ == "__main__":
    main()
