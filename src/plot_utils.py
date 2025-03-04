from datetime import timedelta
from typing import Optional, Union

import pandas as pd
import numpy as np
import plotly.express as px


def plot_aggregated_time_series(
    features: pd.DataFrame,
    targets: Union[pd.Series, np.ndarray],
    row_id: int,
    predictions: Optional[Union[pd.Series, np.ndarray]] = None,
):
    """
    Plots the time series data for a specific location from NYC taxi data.

    Args:
        features (pd.DataFrame): DataFrame containing feature data, including historical ride counts and metadata.
        targets (Union[pd.Series, np.ndarray]): Series or array containing the target values (e.g., actual ride counts).
        row_id (int): Index of the row to plot.
        predictions (Optional[Union[pd.Series, np.ndarray]]): Series or array containing predicted values (optional).

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object showing the time series plot.
    """
    
    if "pickup_location_id" not in features.columns:
        raise ValueError("Column 'pickup_location_id' not found in features DataFrame.")


    if row_id >= len(features) or row_id >= len(targets):
        raise ValueError("row_id is out of bounds for features or targets.")

    location_features = features.iloc[row_id]
    actual_target = targets[row_id] 

    time_series_columns = [
        col for col in features.columns if col.startswith("rides_t-")
    ]
    time_series_values = [location_features[col] for col in time_series_columns] + [
        actual_target
    ]

    # Generate corresponding timestamps for the time series
    time_series_dates = pd.date_range(
        start=location_features["pickup_hour"] - timedelta(hours=len(time_series_columns)),
        end=location_features["pickup_hour"],
        freq="h",
    )

    # Create the plot title with relevant metadata
    title = f"Pickup Hour: {location_features['pickup_hour']}, Location ID: {location_features['pickup_location_id']}"

    # Create the base line plot
    fig = px.line(
        x=time_series_dates,
        y=time_series_values,
        template="plotly_white",
        markers=True,
        title=title,
        labels={"x": "Time", "y": "Ride Counts"},
    )

    # Add the actual target value as a green marker
    fig.add_scatter(
        x=[time_series_dates[-1]],  # Last timestamp
        y=[actual_target],  # Actual target value
        line_color="green",
        mode="markers",
        marker_size=10,
        name="Actual Value",
    )

    # Optionally add the prediction as a red marker
    if predictions is not None:
        if row_id >= len(predictions):
            raise ValueError("row_id is out of bounds for predictions.")
        predicted_value = predictions[row_id]  # Works for both Pandas Series and NumPy arrays
        fig.add_scatter(
            x=[time_series_dates[-1]],  # Last timestamp
            y=[predicted_value],  # Predicted value
            line_color="red",
            mode="markers",
            marker_symbol="x",
            marker_size=15,
            name="Prediction",
        )

    return fig