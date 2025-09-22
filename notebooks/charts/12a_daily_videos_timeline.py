import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

def main():
    # Load data
    script_dir = Path(__file__).parent
    history_path = script_dir / "tbpn_viewing_timeline.csv"

    df = pd.read_csv(history_path)

    # Convert dates properly - Chrome history uses YYYY-MM-DD format
    df['visit_date'] = pd.to_datetime(df['visit_date'], format='%Y-%m-%d')

    # Get the first and last dates
    first_date = df['visit_date'].min()
    last_date = df['visit_date'].max()

    # Create a complete date range from first to last day
    complete_date_range = pd.date_range(start=first_date, end=last_date, freq='D')

    # Group by date to count unique entries per day (not sum of visit_count)
    daily_visits = df.groupby('visit_date').size().reset_index(name='total_visits')

    # Create a complete DataFrame with all dates, filling missing dates with 0
    complete_daily_visits = pd.DataFrame({'visit_date': complete_date_range})

    # Merge with actual data
    complete_daily_visits = complete_daily_visits.merge(
        daily_visits,
        on='visit_date',
        how='left'
    )

    # Fill missing values with 0 and rename the column
    complete_daily_visits['total_visits'] = complete_daily_visits['total_visits'].fillna(0).astype(int)

    # Create custom date labels like the other charts (Month Day format)
    def format_date_label(date):
        return date.strftime('%m/%d')

    # Calculate cumulative video entries over time (no resets)
    complete_daily_visits['total_cumulative'] = complete_daily_visits['total_visits'].cumsum()

    # Create date labels for ALL dates but only show labels for every few days
    complete_daily_visits['date_label'] = complete_daily_visits['visit_date'].apply(format_date_label)

    # Select every Nth date for labels to avoid overcrowding (aim for ~12 labels total)
    total_dates = len(complete_daily_visits)
    if total_dates <= 12:
        # Show all labels if there are 12 or fewer dates
        selected_indices = list(range(total_dates))
    else:
        # Calculate interval to get approximately 12 labels
        interval = max(1, total_dates // 12)
        selected_indices = list(range(0, total_dates, interval))

        # Ensure we don't exceed 12 labels and include the last date
        if len(selected_indices) > 12:
            selected_indices = list(range(0, total_dates, max(1, total_dates // 12)))
        if total_dates - 1 not in selected_indices:
            selected_indices.append(total_dates - 1)

    # Get the selected dates and labels
    selected_dates = complete_daily_visits['visit_date'].iloc[selected_indices]
    selected_labels = complete_daily_visits['date_label'].iloc[selected_indices]

    # Create cumulative line chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=complete_daily_visits['visit_date'],
        y=complete_daily_visits['total_cumulative'],
        mode='lines',
        name='Cumulative Video Entries',
        line=dict(color='black', width=1)
    ))

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Cumulative Gong Hits Listened",
        template='plotly_white',
        font=dict(family="monotype bembo", size=14, color="black"),
        showlegend=False,
        width=647.2,
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),  # Reduced bottom margin to bring x-axis labels closer
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    # Format x-axis with selected labels (every few days to avoid overcrowding)
    fig.update_xaxes(
        showgrid=False,
        showline=False,  # Hide default x-axis line since we'll use zeroline
        linecolor="black",
        tickfont=dict(size=12, family="monotype bembo", color="black"),
        tickangle=0,
        ticktext=selected_labels,
        tickvals=selected_dates,
        side="bottom"  # Keep ticks at bottom
    )

    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linecolor="black",
        tickfont=dict(size=12, family="monotype bembo", color="black"),
        range=[0, None],  # Start y-axis at 0 to align with x-axis
        rangemode='tozero',  # Ensure range starts exactly at 0
        zeroline=True,    # Show line at y=0
        zerolinecolor="black",
        zerolinewidth=1,
        tickmode='array',
        tickvals=[200, 400, 600, 800, 1000, 1200],
        ticktext=["200", "400", "600", "800", "1000", "1200"]
    )

    # Add simple annotations that can be positioned independently
    annotations = [
        dict(
            x=0.1, y=0.075,
            xref="paper", yref="paper",
            text="First Test Run",
            showarrow=False,
            font=dict(family="gill sans", size=12, color="black"),
        ),
        dict(
            x=0.27, y=0.10,
            xref="paper", yref="paper",
            text="Detection Pipeline<br>Development",
            showarrow=False,
            font=dict(family="gill sans", size=12, color="black"),
        ),
        dict(
            x=0.51, y=0.145,
            xref="paper", yref="paper",
            text="Loudness<br>Testing",
            showarrow=False,
            font=dict(family="gill sans", size=12, color="black"),
        ),
        dict(
            x=0.78, y=0.25,
            xref="paper", yref="paper",
            text="Manual Labeling<br>Begins",
            showarrow=False,
            font=dict(family="gill sans", size=12, color="black"),
        ),
        dict(
            x=0.82, y=0.67,
            xref="paper", yref="paper",
            text="Trough of<br>Disillusionment",
            showarrow=False,
            font=dict(family="gill sans", size=12, color="black"),
        ),
        dict(
            x=0.99, y=.99,
            xref="paper", yref="paper",
            text="Analysis Begins",
            showarrow=False,
            font=dict(family="gill sans", size=12, color="black"),
        )
    ]

    fig.update_layout(annotations=annotations)

    # Save chart
    output_dir = script_dir / "charts_output"
    output_dir.mkdir(exist_ok=True)
    fig.write_image(output_dir / "12a_daily_videos_timeline.png", width=647.2, height=400, scale=2)

if __name__ == "__main__":
    main()
