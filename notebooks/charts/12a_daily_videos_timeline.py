import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

def main():
    # Load data
    script_dir = Path(__file__).parent
    history_path = script_dir.parent.parent / "data" / "chrome_history" / "tbpn_viewing_timeline.csv"

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
        line=dict(color='black', width=1.5)
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Total Video Entries",
        template='plotly_white',
        font=dict(family="monotype bembo", size=14, color="black"),
        showlegend=False,
        width=647.2,
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    # Format x-axis with selected labels (every few days to avoid overcrowding)
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor="black",
        tickfont=dict(size=12, family="monotype bembo", color="black"),
        tickangle=0,
        ticktext=selected_labels,
        tickvals=selected_dates
    )

    fig.update_yaxes(showgrid=False, showline=True, linecolor="black", tickfont=dict(size=12, family="monotype bembo", color="black"))

    # Save chart
    output_dir = script_dir / "charts_output"
    output_dir.mkdir(exist_ok=True)
    fig.write_image(output_dir / "12a_daily_videos_timeline.png", width=647.2, height=400, scale=2)

if __name__ == "__main__":
    main()
