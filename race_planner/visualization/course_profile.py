"""
Visualization module for race course profiles and analysis.
"""

from pathlib import Path
from typing import List, Dict, Optional

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
        df_gradient["gradient_pct"] = df_gradient["grade"]

        # Create figure
        fig = go.Figure()

        # Add elevation profile trace with fill
        fig.add_trace(
            go.Scatter(
                x=df_gradient["cum_dist_m"] / 1000,  # Convert to km
                y=df_gradient["ele_m"],
                mode="lines",
                name="Elevation",
                line=dict(color="#2563eb", width=2.5),  # Sleek blue
                fill="tozeroy",  # Fill area under curve
                fillcolor="rgba(37, 99, 235, 0.12)",  # Light blue with transparency
                customdata=list(
                    zip(df_gradient["cum_ele_gain_m"], df_gradient["gradient_pct"])
                ),  # Add cumulative gain and gradient
                hovertemplate=(
                    '<b style="color:#92400e">Distance:</b> <span style="color:#92400e">%{x:.2f} km</span><br>'
                    '<b style="color:#92400e">Elevation:</b> <span style="color:#92400e">%{y:.0f} m</span><br>'
                    '<b style="color:#92400e">Gradient:</b> <span style="color:#92400e">%{customdata[1]:.1f}%</span><br>'
                    '<b style="color:#92400e">Accum. Elevation Gain:</b> <span style="color:#92400e">%{customdata[0]:.0f} m</span><br>'
                    "<extra></extra>"
                ),
            )
        )

        # Add aid station markers
        for aid in self.aid_stations:
            name = aid.get("name", "Unknown")
            jap_name = aid.get("jap_name", "")
            distance_km = aid.get("distance_km", 0)
            notes = aid.get("notes", "")
            gmaps_link = aid.get("gmaps_link", "")

            # Get point data from course
            distance_m = distance_km * 1000
            point = self.course.get_point_at_distance(distance_m)
            elevation_m = float(point["ele_m"])
            cum_gain_m = float(point["cum_ele_gain_m"])

            # Format name with Japanese
            full_name = f"{name} ({jap_name})" if jap_name else name

            # Create detailed click text (shows on click, not hover)
            click_text = (
                f"<b>{full_name}</b><br>"
                f"<b>Distance:</b> {distance_km:.1f} km<br>"
                f"<b>Elevation:</b> {elevation_m:.0f} m<br>"
                f"<b>Cumulative Gain:</b> {cum_gain_m:.0f} m<br>"
            )
            if notes:
                click_text += f"<b>Notes:</b> {notes}<br>"
            if gmaps_link:
                click_text += (
                    f"<b>Google Maps:</b> <a href='{gmaps_link}' target='_blank'>Open</a><br>"
                )

            # Add marker for aid station
            fig.add_trace(
                go.Scatter(
                    x=[distance_km],
                    y=[elevation_m],
                    mode="markers",
                    name=name,
                    marker=dict(
                        size=16,
                        color="#f59e0b",  # Sleek gold/amber
                        symbol="diamond",
                        line=dict(width=2, color="white"),
                    ),
                    customdata=[[click_text]],
                    hovertemplate="<extra></extra>",  # Make clickable but show nothing on hover
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
                bgcolor="rgba(255, 251, 235, 0.75)",  # Light gold background
                bordercolor="#f59e0b",  # Gold border
                borderwidth=1,
                borderpad=4,
                font=dict(
                    size=11,
                    color="#78350f",  # Dark gold text
                    family="Inter, system-ui, sans-serif",
                    weight=600,
                ),
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(
                    size=24,
                    color="#1e3a8a",  # Deep blue
                    family="Inter, system-ui, sans-serif",
                    weight=600,
                ),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title=dict(
                    text="Distance (km)",
                    font=dict(
                        size=16,
                        family="Inter, system-ui, sans-serif",
                        weight=500,
                        color="#1e40af",
                    ),
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor="#dbeafe",  # Light blue grid
                minor=dict(
                    showgrid=True,
                    gridcolor="#bfdbfe",
                    gridwidth=0.5,
                    griddash="dot",
                    ticklen=4,
                ),
                zeroline=False,
                tickfont=dict(size=11, family="Inter, system-ui, sans-serif", color="#374151"),
                minor_ticks="inside",
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikedash="solid",
                spikecolor="#94a3b8",
                spikethickness=1,
            ),
            yaxis=dict(
                title=dict(
                    text="Elevation (m)",
                    font=dict(
                        size=16,
                        family="Inter, system-ui, sans-serif",
                        weight=500,
                        color="#1e40af",
                    ),
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor="#dbeafe",  # Light blue grid
                minor=dict(
                    showgrid=True,
                    gridcolor="#bfdbfe",
                    gridwidth=0.5,
                    griddash="dot",
                    ticklen=4,
                ),
                zeroline=False,
                tickfont=dict(size=11, family="Inter, system-ui, sans-serif", color="#374151"),
                minor_ticks="inside",
            ),
            hovermode="closest",  # Show hover without unified x-label
            hoverlabel=dict(
                bgcolor="#fffbeb",  # Light gold background
                font_size=14,
                font_family="Inter, system-ui, sans-serif",
                bordercolor="#f59e0b",  # Gold border
                namelength=0,  # Hide trace names
            ),
            clickmode="event",  # Enable click events without selection
            template="plotly_white",
            height=600,
            showlegend=False,
            plot_bgcolor="#f8fafc",  # Very light blue-gray
            paper_bgcolor="white",
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
            font=dict(size=13, family="Inter, system-ui, sans-serif", color="#1e3a8a"),
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
                    var xaxis = gd._fullLayout.xaxis;
                    var yaxis = gd._fullLayout.yaxis;
                    
                    // If clicked on elevation line, try to find nearby aid station
                    if (clickedPoint.curveNumber === 0) {
                        var clickX = clickedPoint.x;
                        var clickY = clickedPoint.y;
                        
                        // Larger tolerance for mobile - 60 pixel radius
                        var tolerancePixels = 60;
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
                            clickedPoint = { 
                                customdata: [nearest.customdata],
                                x: nearest.x,
                                y: nearest.y
                            };
                        } else {
                            return; // No nearby aid station, ignore click
                        }
                    } else if (clickedPoint.curveNumber > 0) {
                        // Direct click on aid station marker
                    } else {
                        return;
                    }
                    
                    // Remove existing info div if any
                    if (infoDiv) {
                        infoDiv.remove();
                    }
                    
                    // Get plot container position and dimensions
                    var plotBounds = gd.getBoundingClientRect();
                    
                    // Convert data coordinates to pixel coordinates relative to plot
                    var xaxis = gd._fullLayout.xaxis;
                    var yaxis = gd._fullLayout.yaxis;
                    var l = gd._fullLayout.margin.l;
                    var t = gd._fullLayout.margin.t;
                    
                    // Calculate pixel position within the plot area
                    var plotAreaWidth = plotBounds.width - l - gd._fullLayout.margin.r;
                    var plotAreaHeight = plotBounds.height - t - gd._fullLayout.margin.b;
                    
                    var xFraction = (clickedPoint.x - xaxis.range[0]) / (xaxis.range[1] - xaxis.range[0]);
                    var yFraction = (yaxis.range[1] - clickedPoint.y) / (yaxis.range[1] - yaxis.range[0]);
                    
                    var pixelX = l + xFraction * plotAreaWidth;
                    var pixelY = t + yFraction * plotAreaHeight;
                    
                    // Create info popup with fixed positioning for better mobile behavior
                    infoDiv = document.createElement('div');
                    infoDiv.style.position = 'fixed';
                    infoDiv.style.backgroundColor = 'rgba(255, 251, 235, 0.98)';
                    infoDiv.style.border = '2px solid #f59e0b';
                    infoDiv.style.borderRadius = '8px';
                    infoDiv.style.padding = '14px 18px';
                    infoDiv.style.fontFamily = 'Inter, system-ui, sans-serif';
                    infoDiv.style.fontSize = '14px';
                    infoDiv.style.color = '#78350f';
                    infoDiv.style.boxShadow = '0 6px 12px rgba(0, 0, 0, 0.15)';
                    infoDiv.style.zIndex = '10000';
                    infoDiv.style.maxWidth = '90vw';
                    infoDiv.style.maxHeight = '80vh';
                    infoDiv.style.overflow = 'auto';
                    infoDiv.innerHTML = clickedPoint.customdata[0];
                    
                    // Add close button
                    var closeBtn = document.createElement('span');
                    closeBtn.innerHTML = '×';
                    closeBtn.style.position = 'absolute';
                    closeBtn.style.right = '8px';
                    closeBtn.style.top = '4px';
                    closeBtn.style.cursor = 'pointer';
                    closeBtn.style.fontSize = '24px';
                    closeBtn.style.fontWeight = 'bold';
                    closeBtn.style.color = '#f59e0b';
                    closeBtn.style.lineHeight = '1';
                    closeBtn.style.padding = '4px 8px';
                    closeBtn.onclick = function(e) {
                        e.stopPropagation();
                        infoDiv.remove();
                        infoDiv = null;
                    };
                    infoDiv.appendChild(closeBtn);
                    
                    document.body.appendChild(infoDiv);
                    
                    // Position popup after adding to DOM to get actual dimensions
                    var infoWidth = infoDiv.offsetWidth;
                    var infoHeight = infoDiv.offsetHeight;
                    var viewportWidth = window.innerWidth;
                    var viewportHeight = window.innerHeight;
                    
                    // Calculate position relative to viewport
                    var left = plotBounds.left + pixelX + 10;
                    var top = plotBounds.top + pixelY - 10;
                    
                    // Keep popup within viewport bounds
                    if (left + infoWidth > viewportWidth - 10) {
                        left = plotBounds.left + pixelX - infoWidth - 10;
                    }
                    if (left < 10) {
                        left = 10;
                    }
                    
                    if (top + infoHeight > viewportHeight - 10) {
                        top = viewportHeight - infoHeight - 10;
                    }
                    if (top < 10) {
                        top = 10;
                    }
                    
                    infoDiv.style.left = left + 'px';
                    infoDiv.style.top = top + 'px';
                });
                
                // Close on click outside (with delay to prevent immediate closing)
                setTimeout(function() {
                    document.addEventListener('click', function(e) {
                        if (infoDiv && !infoDiv.contains(e.target) && !e.target.closest('.plotly')) {
                            infoDiv.remove();
                            infoDiv = null;
                        }
                    });
                }, 100);
            });
            </script>
            """

            # Write HTML with custom JavaScript
            html_str = fig.to_html(include_plotlyjs="cdn")
            html_str = html_str.replace("</body>", click_js + "</body>")

            with open(output_path, "w", encoding="utf-8") as f:
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
