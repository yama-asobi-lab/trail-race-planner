"""
Visualization module for race course profiles and analysis.
"""

from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from race_planner.course import Course


class CourseProfilePlotter:
    """
    Creates interactive elevation profile plots for race courses.

    Args:
        course: Course object with loaded GPX data.
        aid_stations: List of aid station configurations from YAML.
    """

    def __init__(self, course: Course, aid_stations: List[Dict]):
        """Initialize plotter with course and aid station data."""
        self.course = course
        self.aid_stations = aid_stations

    def create_elevation_profile(
        self,
        output_path: Optional[Path | str] = None,
        title: str = "Course Elevation Profile",
    ) -> go.Figure:
        """
        Create an interactive elevation profile plot.

        Args:
            output_path: Path to save HTML file (optional).
            title: Plot title.

        Returns:
            Plotly figure object.
        """
        # Get course data
        df = self.course.df

        # Create figure
        fig = go.Figure()

        # Add elevation profile trace with fill
        fig.add_trace(
            go.Scatter(
                x=df['cum_dist_m'] / 1000,  # Convert to km
                y=df['ele_m'],
                mode='lines',
                name='Elevation',
                line=dict(color='#1f77b4', width=2.5),
                fill='tozeroy',  # Fill area under curve
                fillcolor='rgba(31, 119, 180, 0.15)',  # Light blue with transparency
                customdata=df['cum_ele_gain_m'],  # Add cumulative gain data
                hovertemplate=(
                    '<b>Distance:</b> %{x:.2f} km<br>'
                    '<b>Elevation:</b> %{y:.0f} m<br>'
                    '<b>Accum. Elevation Gain:</b> %{customdata:.0f} m<br>'
                    '<extra></extra>'
                ),
            )
        )

        # Add aid station markers
        for aid in self.aid_stations:
            name = aid.get('name', 'Unknown')
            jap_name = aid.get('jap_name', '')
            distance_km = aid.get('distance_km', 0)
            notes = aid.get('notes', '')
            stop_time_s = aid.get('stop_time_s', 0)

            # Get point data from course
            distance_m = distance_km * 1000
            point = self.course.get_point_at_distance(distance_m)
            elevation_m = float(point['ele_m'])
            cum_gain_m = float(point['cum_ele_gain_m'])
            cum_loss_m = float(point['cum_ele_loss_m'])

            # Format name with Japanese
            full_name = f"{name} ({jap_name})" if jap_name else name

            # Create detailed hover text
            hover_text = (
                f"<b>{full_name}</b><br>"
                f"<b>Distance:</b> {distance_km:.1f} km<br>"
                f"<b>Elevation:</b> {elevation_m:.0f} m<br>"
                f"<b>Cumulative Gain:</b> {cum_gain_m:.0f} m<br>"
                f"<b>Cumulative Loss:</b> {cum_loss_m:.0f} m<br>"
                f"<b>Stop Time:</b> {stop_time_s}s<br>"
            )
            if notes:
                hover_text += f"<b>Notes:</b> {notes}<br>"

            # Add marker for aid station
            fig.add_trace(
                go.Scatter(
                    x=[distance_km],
                    y=[elevation_m],
                    mode='markers',
                    name=name,
                    marker=dict(
                        size=12,
                        color='#e74c3c',  # Modern red
                        symbol='triangle-up',
                        line=dict(width=2, color='white'),
                    ),
                    hovertemplate=hover_text + '<extra></extra>',
                    showlegend=False,
                )
            )

            # Add text label with background
            fig.add_annotation(
                x=distance_km,
                y=elevation_m,
                text=name,
                showarrow=False,
                yshift=15,  # Position above marker
                bgcolor='rgba(255, 255, 255, 0.7)',  # White with 85% opacity
                bordercolor='#e0e0e0',
                borderwidth=1,
                borderpad=4,
                font=dict(
                    size=10,
                    color='#2c3e50',
                    family='Inter, system-ui, sans-serif',
                    weight=600,
                ),
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(
                    size=24,
                    color='#2c3e50',
                    family='Inter, system-ui, sans-serif',
                    weight=600,
                ),
                x=0.5,
                xanchor='center',
            ),
            xaxis=dict(
                title=dict(
                    text='Distance (km)',
                    font=dict(
                        size=14, family='Inter, system-ui, sans-serif', weight=500
                    ),
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor='#e0e0e0',
                minor=dict(showgrid=True, gridcolor='#f5f5f5', griddash='dot'),
                zeroline=False,
                tickfont=dict(size=11, family='Inter, system-ui, sans-serif'),
            ),
            yaxis=dict(
                title=dict(
                    text='Elevation (m)',
                    font=dict(
                        size=14, family='Inter, system-ui, sans-serif', weight=500
                    ),
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor='#e0e0e0',
                minor=dict(showgrid=True, gridcolor='#f5f5f5', griddash='dot'),
                zeroline=False,
                tickfont=dict(size=11, family='Inter, system-ui, sans-serif'),
            ),
            hovermode='x unified',  # Vertical line across plot
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Inter, system-ui, sans-serif',
                bordercolor='#ddd',
            ),
            template='plotly_white',
            height=600,
            showlegend=False,
            plot_bgcolor='#fafafa',
            paper_bgcolor='white',
            margin=dict(l=80, r=40, t=80, b=60),
        )

        # Add elevation statistics annotation
        total_gain = self.course.total_elevation_gain_m
        total_loss = self.course.total_elevation_loss_m
        total_distance = self.course.total_distance_km

        annotation_text = (
            f"Total Distance: {total_distance:.1f} km<br>"
            f"Total Gain: {total_gain:.0f} m<br>"
            f"Total Loss: {total_loss:.0f} m"
        )

        fig.add_annotation(
            text=annotation_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#ddd",
            borderwidth=1.5,
            borderpad=10,
            align="left",
            xanchor="left",
            yanchor="top",
            font=dict(size=12, family='Inter, system-ui, sans-serif', color='#2c3e50'),
        )

        # Save to file if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            logger.success(f"Elevation profile saved to: {output_path}")

        return fig

    def show(self, title: str = "Course Elevation Profile") -> None:
        """
        Display the elevation profile in a browser.

        Args:
            title: Plot title.
        """
        fig = self.create_elevation_profile(title=title)
        fig.show()


def plot_course_profile(
    course: Course,
    aid_stations: List[Dict],
    output_path: Path | str,
    title: str = "Course Elevation Profile",
) -> go.Figure:
    """
    Convenience function to create and save an elevation profile plot.

    Args:
        course: Course object.
        aid_stations: List of aid station configurations.
        output_path: Path to save HTML file.
        title: Plot title.

    Returns:
        Plotly figure object.
    """
    plotter = CourseProfilePlotter(course, aid_stations)
    return plotter.create_elevation_profile(output_path=output_path, title=title)
