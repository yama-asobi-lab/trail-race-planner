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

        # Calculate gradient percentage for hover (already in percent in 'grade' column)
        df_gradient = df.copy()
        df_gradient['gradient_pct'] = df_gradient['grade']

        # Create figure
        fig = go.Figure()

        # Add elevation profile trace with fill
        fig.add_trace(
            go.Scatter(
                x=df_gradient['cum_dist_m'] / 1000,  # Convert to km
                y=df_gradient['ele_m'],
                mode='lines',
                name='Elevation',
                line=dict(color='#2563eb', width=2.5),  # Sleek blue
                fill='tozeroy',  # Fill area under curve
                fillcolor='rgba(37, 99, 235, 0.12)',  # Light blue with transparency
                customdata=list(
                    zip(df_gradient['cum_ele_gain_m'], df_gradient['gradient_pct'])
                ),  # Add cumulative gain and gradient
                hovertemplate=(
                    '<b>Distance:</b> %{x:.2f} km<br>'
                    '<b>Elevation:</b> %{y:.0f} m<br>'
                    '<b>Gradient:</b> %{customdata[1]:.1f}%<br>'
                    '<b>Accum. Elevation Gain:</b> %{customdata[0]:.0f} m<br>'
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

            # Create detailed click text (shows on click, not hover)
            click_text = (
                f"<b>{full_name}</b><br>"
                f"<b>Distance:</b> {distance_km:.1f} km<br>"
                f"<b>Elevation:</b> {elevation_m:.0f} m<br>"
                f"<b>Cumulative Gain:</b> {cum_gain_m:.0f} m<br>"
                f"<b>Cumulative Loss:</b> {cum_loss_m:.0f} m<br>"
                f"<b>Stop Time:</b> {stop_time_s}s<br>"
            )
            if notes:
                click_text += f"<b>Notes:</b> {notes}<br>"

            # Add marker for aid station
            fig.add_trace(
                go.Scatter(
                    x=[distance_km],
                    y=[elevation_m],
                    mode='markers',
                    name=name,
                    marker=dict(
                        size=16,
                        color='#f59e0b',  # Sleek gold/amber
                        symbol='diamond',
                        line=dict(width=2, color='white'),
                    ),
                    customdata=[[click_text]],
                    hovertemplate='<extra></extra>',  # Make clickable but show nothing on hover
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
                bgcolor='rgba(255, 251, 235, 0.75)',  # Light gold background
                bordercolor='#f59e0b',  # Gold border
                borderwidth=1,
                borderpad=4,
                font=dict(
                    size=11,
                    color='#78350f',  # Dark gold text
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
                    color='#1e3a8a',  # Deep blue
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
                        size=16,
                        family='Inter, system-ui, sans-serif',
                        weight=500,
                        color='#1e40af',
                    ),
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor='#dbeafe',  # Light blue grid
                minor=dict(
                    showgrid=True,
                    gridcolor='#bfdbfe',
                    gridwidth=0.5,
                    griddash='dot',
                    ticklen=4,
                ),
                zeroline=False,
                tickfont=dict(
                    size=11, family='Inter, system-ui, sans-serif', color='#374151'
                ),
                minor_ticks="inside",
            ),
            yaxis=dict(
                title=dict(
                    text='Elevation (m)',
                    font=dict(
                        size=16,
                        family='Inter, system-ui, sans-serif',
                        weight=500,
                        color='#1e40af',
                    ),
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor='#dbeafe',  # Light blue grid
                minor=dict(
                    showgrid=True,
                    gridcolor='#bfdbfe',
                    gridwidth=0.5,
                    griddash='dot',
                    ticklen=4,
                ),
                zeroline=False,
                tickfont=dict(
                    size=11, family='Inter, system-ui, sans-serif', color='#374151'
                ),
                minor_ticks="inside",
            ),
            hovermode='x unified',  # Vertical line across plot
            hoverlabel=dict(
                bgcolor='#fffbeb',  # Light gold background
                font_size=14,
                font_family='Inter, system-ui, sans-serif',
                bordercolor='#f59e0b',  # Gold border
            ),
            clickmode='event',  # Enable click events without selection
            template='plotly_white',
            height=600,
            showlegend=False,
            plot_bgcolor='#f8fafc',  # Very light blue-gray
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
            bgcolor="rgba(239, 246, 255, 0.95)",  # Light blue background
            bordercolor="#2563eb",  # Blue border
            borderwidth=1.5,
            borderpad=10,
            align="left",
            xanchor="left",
            yanchor="top",
            font=dict(size=13, family='Inter, system-ui, sans-serif', color='#1e3a8a'),
        )

        # Save to file if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Add custom JavaScript for click-to-show aid station info
            click_js = """
            <script>
            document.addEventListener('DOMContentLoaded', function() {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                var infoDiv = null;
                
                // Store aid station data for proximity detection
                var aidStations = [];
                gd.data.forEach(function(trace, idx) {
                    if (idx > 0) { // Skip first trace (elevation line)
                        aidStations.push({
                            x: trace.x[0],
                            y: trace.y[0],
                            customdata: trace.customdata[0],
                            traceIdx: idx
                        });
                    }
                });
                
                // Click handler with proximity detection
                gd.on('plotly_click', function(data) {
                    var clickedPoint = data.points[0];
                    
                    // If clicked on elevation line, try to find nearby aid station
                    if (clickedPoint.curveNumber === 0) {
                        var xaxis = gd._fullLayout.xaxis;
                        var yaxis = gd._fullLayout.yaxis;
                        var clickX = clickedPoint.x;
                        var clickY = clickedPoint.y;
                        
                        // Convert click tolerance from pixels to data units
                        var tolerancePixels = 40; // 40 pixel radius for mobile-friendly clicking
                        var xRange = xaxis.range[1] - xaxis.range[0];
                        var yRange = yaxis.range[1] - yaxis.range[0];
                        var plotWidth = gd._fullLayout.width;
                        var plotHeight = gd._fullLayout.height;
                        
                        var xTolerance = (tolerancePixels / plotWidth) * xRange;
                        var yTolerance = (tolerancePixels / plotHeight) * yRange;
                        
                        // Find nearest aid station within tolerance
                        var nearest = null;
                        var minDist = Infinity;
                        
                        aidStations.forEach(function(station) {
                            var dx = (station.x - clickX) / xTolerance;
                            var dy = (station.y - clickY) / yTolerance;
                            var dist = Math.sqrt(dx * dx + dy * dy);
                            
                            if (dist < 1 && dist < minDist) {
                                minDist = dist;
                                nearest = station;
                            }
                        });
                        
                        if (nearest) {
                            clickedPoint = { customdata: [nearest.customdata] };
                        } else {
                            return; // No nearby aid station, ignore click
                        }
                    } else if (clickedPoint.curveNumber > 0) {
                        // Direct click on aid station marker - data is already in correct format
                        // clickedPoint.customdata is already correct
                    } else {
                        return;
                    }
                    
                    // Remove existing info div if any
                    if (infoDiv) {
                        infoDiv.remove();
                    }
                    
                    // Create info popup
                    infoDiv = document.createElement('div');
                    infoDiv.style.position = 'absolute';
                    infoDiv.style.left = (data.event.pageX + 10) + 'px';
                    infoDiv.style.top = (data.event.pageY - 10) + 'px';
                    infoDiv.style.backgroundColor = 'rgba(255, 251, 235, 0.98)';
                    infoDiv.style.border = '2px solid #f59e0b';
                    infoDiv.style.borderRadius = '6px';
                    infoDiv.style.padding = '12px 16px';
                    infoDiv.style.fontFamily = 'Inter, system-ui, sans-serif';
                    infoDiv.style.fontSize = '13px';
                    infoDiv.style.color = '#78350f';
                    infoDiv.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
                    infoDiv.style.zIndex = '1000';
                    infoDiv.style.maxWidth = '300px';
                    infoDiv.innerHTML = clickedPoint.customdata[0];
                    
                    // Add close button
                    var closeBtn = document.createElement('span');
                    closeBtn.innerHTML = '×';
                    closeBtn.style.position = 'absolute';
                    closeBtn.style.right = '8px';
                    closeBtn.style.top = '4px';
                    closeBtn.style.cursor = 'pointer';
                    closeBtn.style.fontSize = '20px';
                    closeBtn.style.fontWeight = 'bold';
                    closeBtn.style.color = '#f59e0b';
                    closeBtn.onclick = function() {
                        infoDiv.remove();
                        infoDiv = null;
                    };
                    infoDiv.appendChild(closeBtn);
                    
                    document.body.appendChild(infoDiv);
                });
                
                // Close on click outside
                document.addEventListener('click', function(e) {
                    if (infoDiv && !infoDiv.contains(e.target) && !e.target.closest('.plotly')) {
                        infoDiv.remove();
                        infoDiv = null;
                    }
                });
            });
            </script>
            """

            # Write HTML with custom JavaScript
            html_str = fig.to_html(include_plotlyjs='cdn')
            html_str = html_str.replace('</body>', click_js + '</body>')

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_str)

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
