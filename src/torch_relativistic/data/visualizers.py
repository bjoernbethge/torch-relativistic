"""
Visualization tools for space datasets with relativistic effects.
"""

from typing import List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from torch_geometric.data import Data


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    figure_width: int = 1200
    figure_height: int = 800
    color_scheme: str = "viridis"
    background_color: str = "black"
    grid_color: str = "gray"
    text_color: str = "white"
    marker_size: int = 8
    line_width: int = 2
    opacity: float = 0.8
    animation_frame_duration: int = 500  # ms
    show_grid: bool = True
    show_legend: bool = True


class SpaceDataVisualizer:
    """Comprehensive visualization tools for space datasets with relativistic effects."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self._setup_plotly_theme()
    
    def _setup_plotly_theme(self):
        """Setup custom Plotly theme for space visualizations."""
        self.theme_template = {
            'layout': {
                'paper_bgcolor': self.config.background_color,
                'plot_bgcolor': self.config.background_color,
                'font': {'color': self.config.text_color, 'family': 'Arial, sans-serif'},
                'xaxis': {
                    'gridcolor': self.config.grid_color,
                    'showgrid': self.config.show_grid,
                    'zeroline': False
                },
                'yaxis': {
                    'gridcolor': self.config.grid_color,
                    'showgrid': self.config.show_grid,
                    'zeroline': False
                },
                'colorway': px.colors.qualitative.Set3
            }
        }
    
    def plot_satellite_orbits_3d(
        self, 
        df: pl.DataFrame, 
        title: str = "Satellite Orbits in 3D Space",
        color_by: Optional[str] = None,
        animation_column: Optional[str] = None
    ) -> go.Figure:
        """
        Create 3D visualization of satellite orbits.
        
        Args:
            df: DataFrame with satellite position data
            title: Plot title
            color_by: Column to use for coloring satellites
            animation_column: Column to use for animation frames
            
        Returns:
            Plotly Figure object
        """
        # Prepare data
        df_pandas = df.to_pandas()
        
        # Determine color column
        if color_by is None:
            color_by = 'satellite_id' if 'satellite_id' in df.columns else None
        
        # Create 3D scatter plot
        if animation_column and animation_column in df.columns:
            fig = px.scatter_3d(
                df_pandas,
                x='x_km', y='y_km', z='z_km',
                color=color_by,
                animation_frame=animation_column,
                title=title,
                labels={
                    'x_km': 'X Position (km)',
                    'y_km': 'Y Position (km)', 
                    'z_km': 'Z Position (km)'
                },
                opacity=self.config.opacity,
                size_max=self.config.marker_size
            )
        else:
            fig = px.scatter_3d(
                df_pandas,
                x='x_km', y='y_km', z='z_km',
                color=color_by,
                title=title,
                labels={
                    'x_km': 'X Position (km)',
                    'y_km': 'Y Position (km)',
                    'z_km': 'Z Position (km)'
                },
                opacity=self.config.opacity
            )
        
        # Add Earth sphere
        self._add_earth_sphere(fig)
        
        # Update layout
        fig.update_layout(
            **self.theme_template['layout'],
            width=self.config.figure_width,
            height=self.config.figure_height,
            title=title,
            scene=dict(
                xaxis_title="X Position (km)",
                yaxis_title="Y Position (km)",
                zaxis_title="Z Position (km)",
                bgcolor=self.config.background_color,
                xaxis=dict(gridcolor=self.config.grid_color),
                yaxis=dict(gridcolor=self.config.grid_color),
                zaxis=dict(gridcolor=self.config.grid_color)
            )
        )
        
        if animation_column:
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = self.config.animation_frame_duration
        
        return fig
    
    def _add_earth_sphere(self, fig: go.Figure, radius_km: float = 6371.0):
        """Add Earth sphere to 3D plot."""
        # Create sphere coordinates
        phi = np.linspace(0, 2*np.pi, 30)
        theta = np.linspace(0, np.pi, 20)
        phi, theta = np.meshgrid(phi, theta)
        
        x = radius_km * np.sin(theta) * np.cos(phi)
        y = radius_km * np.sin(theta) * np.sin(phi)
        z = radius_km * np.cos(theta)
        
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale='Blues',
            opacity=0.6,
            showscale=False,
            name='Earth'
        ))
    
    def plot_velocity_distribution(
        self, 
        df: pl.DataFrame,
        title: str = "Satellite Velocity Distribution"
    ) -> go.Figure:
        """Plot velocity magnitude distribution."""
        df_pandas = df.to_pandas()
        
        # Calculate velocity magnitudes if not present
        if 'speed_m_s' not in df_pandas.columns:
            if all(col in df_pandas.columns for col in ['vx_km_s', 'vy_km_s', 'vz_km_s']):
                df_pandas['speed_km_s'] = np.sqrt(
                    df_pandas['vx_km_s']**2 + 
                    df_pandas['vy_km_s']**2 + 
                    df_pandas['vz_km_s']**2
                )
            else:
                raise ValueError("Velocity data not found in DataFrame")
        else:
            df_pandas['speed_km_s'] = df_pandas['speed_m_s'] / 1000
        
        # Create histogram
        fig = px.histogram(
            df_pandas,
            x='speed_km_s',
            title=title,
            labels={'speed_km_s': 'Speed (km/s)', 'count': 'Number of Satellites'},
            color_discrete_sequence=[px.colors.qualitative.Set3[0]]
        )
        
        # Update layout
        fig.update_layout(
            **self.theme_template['layout'],
            width=self.config.figure_width,
            height=self.config.figure_height//2,
            title=title
        )
        
        return fig
    
    def plot_relativistic_effects(
        self, 
        df: pl.DataFrame,
        title: str = "Relativistic Effects Analysis"
    ) -> go.Figure:
        """Plot relativistic effects like time dilation and Lorentz factors."""
        df_pandas = df.to_pandas()
        
        # Calculate relativistic factors if not present
        c = 299792458.0  # m/s
        
        if 'speed_m_s' not in df_pandas.columns:
            if all(col in df_pandas.columns for col in ['vx_km_s', 'vy_km_s', 'vz_km_s']):
                df_pandas['speed_m_s'] = 1000 * np.sqrt(
                    df_pandas['vx_km_s']**2 + 
                    df_pandas['vy_km_s']**2 + 
                    df_pandas['vz_km_s']**2
                )
            else:
                raise ValueError("Velocity data not found")
        
        # Calculate relativistic parameters
        df_pandas['beta'] = df_pandas['speed_m_s'] / c
        df_pandas['gamma'] = 1.0 / np.sqrt(1.0 - df_pandas['beta']**2)
        df_pandas['time_dilation'] = df_pandas['gamma'] - 1.0  # Fractional time dilation
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Beta (v/c)', 'Lorentz Factor (γ)', 'Time Dilation (Δt/t)', 'Speed Distribution'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Beta distribution
        fig.add_trace(
            go.Histogram(x=df_pandas['beta'], name='β = v/c', marker_color=px.colors.qualitative.Set3[0]),
            row=1, col=1
        )
        
        # Gamma distribution
        fig.add_trace(
            go.Histogram(x=df_pandas['gamma'], name='γ', marker_color=px.colors.qualitative.Set3[1]),
            row=1, col=2
        )
        
        # Time dilation
        fig.add_trace(
            go.Histogram(x=df_pandas['time_dilation'] * 1e9, name='Δt/t (ns/s)', 
                        marker_color=px.colors.qualitative.Set3[2]),
            row=2, col=1
        )
        
        # Speed vs time dilation scatter
        if 'datetime' in df_pandas.columns:
            fig.add_trace(
                go.Scatter(x=df_pandas['speed_m_s']/1000, y=df_pandas['time_dilation']*1e9,
                          mode='markers', name='Speed vs Time Dilation',
                          marker=dict(color=px.colors.qualitative.Set3[3])),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            **self.theme_template['layout'],
            width=self.config.figure_width,
            height=self.config.figure_height,
            title_text=title,
            showlegend=False
        )
        
        return fig
    
    def plot_orbital_elements(
        self, 
        df: pl.DataFrame,
        title: str = "Orbital Elements Analysis"
    ) -> go.Figure:
        """Plot orbital elements like inclination, eccentricity, etc."""
        if not any(col in df.columns for col in ['inclination_deg', 'eccentricity', 'mean_motion', 'raan_deg']):
            raise ValueError("Orbital elements not found in DataFrame")
        
        df_pandas = df.to_pandas()
        
        # Calculate altitude from mean motion if available
        if 'mean_motion' in df_pandas.columns and 'altitude_km' not in df_pandas.columns:
            # Convert mean motion (revolutions per day) to altitude
            # Using simplified formula: n = sqrt(GM/a^3) where n is mean motion
            mu_earth = 3.986004418e14  # m^3/s^2
            
            # Convert revolutions/day to radians/second
            mean_motion_rad_s = df_pandas['mean_motion'] * 2 * np.pi / 86400
            
            # Calculate semi-major axis: a = (GM/n^2)^(1/3)
            semi_major_axis_m = (mu_earth / (mean_motion_rad_s**2))**(1/3)
            
            # Convert to altitude (subtract Earth's radius)
            earth_radius_km = 6371.0
            df_pandas['altitude_km'] = (semi_major_axis_m / 1000) - earth_radius_km
        
        # Create subplots for different orbital elements
        subplot_titles = []
        has_inclination = 'inclination_deg' in df_pandas.columns
        has_eccentricity = 'eccentricity' in df_pandas.columns  
        has_altitude = 'altitude_km' in df_pandas.columns
        has_raan = 'raan_deg' in df_pandas.columns
        
        # Determine which plots to show
        plots_config = []
        if has_inclination:
            plots_config.append(('inclination_deg', 'Inclination Distribution'))
        if has_eccentricity:
            plots_config.append(('eccentricity', 'Eccentricity Distribution'))
        if has_altitude:
            plots_config.append(('altitude_km', 'Altitude Distribution'))
        if has_raan:
            plots_config.append(('raan_deg', 'RAAN Distribution'))
        
        # Create subplot grid based on available data
        if len(plots_config) >= 4:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[config[1] for config in plots_config[:4]],
                specs=[[{"type": "histogram"}, {"type": "histogram"}],
                       [{"type": "histogram"}, {"type": "scatter"}]]
            )
        elif len(plots_config) == 3:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[plots_config[0][1], plots_config[1][1], plots_config[2][1], 'Inclination vs Eccentricity'],
                specs=[[{"type": "histogram"}, {"type": "histogram"}],
                       [{"type": "histogram"}, {"type": "scatter"}]]
            )
        else:
            fig = make_subplots(
                rows=1, cols=len(plots_config),
                subplot_titles=[config[1] for config in plots_config]
            )
        
        # Add histogram plots
        colors = px.colors.qualitative.Set3
        
        if len(plots_config) >= 4:
            # 2x2 grid
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            for i, (col_name, title) in enumerate(plots_config[:3]):
                row, col = positions[i]
                if col_name in df_pandas.columns:
                    fig.add_trace(
                        go.Histogram(x=df_pandas[col_name], name=title,
                                   marker_color=colors[i % len(colors)]),
                        row=row, col=col
                    )
            
            # Add scatter plot in position 4
            if has_inclination and has_eccentricity:
                fig.add_trace(
                    go.Scatter(x=df_pandas['inclination_deg'], y=df_pandas['eccentricity'],
                              mode='markers', name='Inc vs Ecc',
                              marker=dict(color=colors[3 % len(colors)])),
                    row=2, col=2
                )
        else:
            # Single row
            for i, (col_name, title) in enumerate(plots_config):
                if col_name in df_pandas.columns:
                    fig.add_trace(
                        go.Histogram(x=df_pandas[col_name], name=title,
                                   marker_color=colors[i % len(colors)]),
                        row=1, col=i+1
                    )
        
        fig.update_layout(
            **self.theme_template['layout'],
            width=self.config.figure_width,
            height=self.config.figure_height,
            title_text=title,
            showlegend=False
        )
        
        return fig
    
    def plot_graph_structure(
        self, 
        data: Data,
        title: str = "Graph Network Structure",
        layout_type: str = "3d"
    ) -> go.Figure:
        """Visualize PyTorch Geometric graph structure."""
        if data.edge_index is None:
            raise ValueError("Graph has no edges to visualize")
        
        edge_index = data.edge_index.numpy()
        
        if hasattr(data, 'pos') and data.pos is not None:
            pos = data.pos.numpy()
        else:
            # Generate random positions if not available
            num_nodes = data.x.shape[0] if data.x is not None else edge_index.max() + 1
            pos = np.random.randn(num_nodes, 3) * 1000  # Random positions
        
        if layout_type == "3d" and pos.shape[1] >= 3:
            return self._plot_3d_graph(edge_index, pos, title)
        else:
            return self._plot_2d_graph(edge_index, pos[:, :2], title)
    
    def _plot_3d_graph(self, edge_index: np.ndarray, pos: np.ndarray, title: str) -> go.Figure:
        """Plot 3D graph structure."""
        fig = go.Figure()
        
        # Add edges
        edge_x, edge_y, edge_z = [], [], []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_x.extend([pos[src, 0], pos[dst, 0], None])
            edge_y.extend([pos[src, 1], pos[dst, 1], None])
            edge_z.extend([pos[src, 2], pos[dst, 2], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(125,125,125,0.5)', width=1),
            name='Edges',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode='markers',
            marker=dict(
                size=self.config.marker_size,
                color=px.colors.qualitative.Set3[0],
                opacity=self.config.opacity
            ),
            name='Nodes'
        ))
        
        fig.update_layout(
            **self.theme_template['layout'],
            width=self.config.figure_width,
            height=self.config.figure_height,
            title=title,
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Z Position",
                bgcolor=self.config.background_color
            )
        )
        
        return fig
    
    def _plot_2d_graph(self, edge_index: np.ndarray, pos: np.ndarray, title: str) -> go.Figure:
        """Plot 2D graph structure."""
        fig = go.Figure()
        
        # Add edges
        edge_x, edge_y = [], []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_x.extend([pos[src, 0], pos[dst, 0], None])
            edge_y.extend([pos[src, 1], pos[dst, 1], None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(color='rgba(125,125,125,0.5)', width=1),
            name='Edges',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=pos[:, 0], y=pos[:, 1],
            mode='markers',
            marker=dict(
                size=self.config.marker_size,
                color=px.colors.qualitative.Set3[0],
                opacity=self.config.opacity
            ),
            name='Nodes'
        ))
        
        fig.update_layout(
            **self.theme_template['layout'],
            width=self.config.figure_width,
            height=self.config.figure_height,
            title=title
        )
        
        return fig
    
    def plot_time_series(
        self, 
        df: pl.DataFrame,
        y_columns: List[str],
        title: str = "Time Series Analysis",
        group_by: Optional[str] = None
    ) -> go.Figure:
        """Plot time series data."""
        if 'datetime' not in df.columns:
            raise ValueError("DateTime column required for time series plot")
        
        df_pandas = df.to_pandas()
        df_pandas['datetime'] = pd.to_datetime(df_pandas['datetime'])
        
        fig = go.Figure()
        
        if group_by and group_by in df_pandas.columns:
            # Plot multiple series
            groups = df_pandas[group_by].unique()
            colors = px.colors.qualitative.Set3
            
            for i, group in enumerate(groups):
                group_data = df_pandas[df_pandas[group_by] == group]
                color = colors[i % len(colors)]
                
                for j, col in enumerate(y_columns):
                    fig.add_trace(go.Scatter(
                        x=group_data['datetime'],
                        y=group_data[col],
                        mode='lines',
                        name=f"{group} - {col}",
                        line=dict(color=color, width=self.config.line_width),
                        opacity=self.config.opacity
                    ))
        else:
            # Single series per column
            for i, col in enumerate(y_columns):
                fig.add_trace(go.Scatter(
                    x=df_pandas['datetime'],
                    y=df_pandas[col],
                    mode='lines',
                    name=col,
                    line=dict(width=self.config.line_width),
                    opacity=self.config.opacity
                ))
        
        fig.update_layout(
            **self.theme_template['layout'],
            width=self.config.figure_width,
            height=self.config.figure_height,
            title=title,
            xaxis_title="Time",
            yaxis_title="Value"
        )
        
        return fig
    
    def create_dashboard(
        self, 
        df: pl.DataFrame,
        save_path: Optional[Path] = None
    ) -> str:
        """Create comprehensive dashboard with multiple visualizations."""
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Space Data Visualization Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    background-color: {self.config.background_color};
                    color: {self.config.text_color};
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                }}
                .dashboard-container {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                .plot-container {{
                    background-color: rgba(255,255,255,0.05);
                    padding: 10px;
                    border-radius: 8px;
                }}
                h1 {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .full-width {{
                    grid-column: 1 / -1;
                }}
            </style>
        </head>
        <body>
            <h1>Space Data Visualization Dashboard</h1>
            <div class="dashboard-container">
        """
        
        # Generate plots
        plots = []
        
        try:
            # 3D Orbits
            fig_3d = self.plot_satellite_orbits_3d(df, "Satellite Orbits")
            plots.append(("3d_orbits", fig_3d.to_html(include_plotlyjs=False, div_id="plot_3d")))
        except Exception as e:
            print(f"Warning: Could not create 3D orbit plot: {e}")
        
        try:
            # Velocity distribution
            fig_vel = self.plot_velocity_distribution(df, "Velocity Distribution")
            plots.append(("velocity", fig_vel.to_html(include_plotlyjs=False, div_id="plot_vel")))
        except Exception as e:
            print(f"Warning: Could not create velocity plot: {e}")
        
        try:
            # Relativistic effects
            fig_rel = self.plot_relativistic_effects(df, "Relativistic Effects")
            plots.append(("relativistic", fig_rel.to_html(include_plotlyjs=False, div_id="plot_rel")))
        except Exception as e:
            print(f"Warning: Could not create relativistic effects plot: {e}")
        
        try:
            # Orbital elements
            fig_orbital = self.plot_orbital_elements(df, "Orbital Elements")
            plots.append(("orbital", fig_orbital.to_html(include_plotlyjs=False, div_id="plot_orbital")))
        except Exception as e:
            print(f"Warning: Could not create orbital elements plot: {e}")
        
        # Add plots to HTML
        for i, (name, plot_html) in enumerate(plots):
            css_class = "plot-container full-width" if i == 0 else "plot-container"
            dashboard_html += f'<div class="{css_class}">{plot_html}</div>\n'
        
        dashboard_html += """
            </div>
        </body>
        </html>
        """
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            print(f"Dashboard saved to {save_path}")
        
        return dashboard_html
    
    def save_figure(self, fig: go.Figure, filepath: Path, format: str = "html"):
        """Save figure to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "html":
            fig.write_html(str(filepath))
        elif format.lower() == "png":
            fig.write_image(str(filepath))
        elif format.lower() == "pdf":
            fig.write_image(str(filepath))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Figure saved to {filepath}")


# Convenience functions for quick plotting
def quick_plot_orbits(df: pl.DataFrame, **kwargs) -> go.Figure:
    """Quick 3D orbit visualization."""
    visualizer = SpaceDataVisualizer()
    return visualizer.plot_satellite_orbits_3d(df, **kwargs)


def quick_plot_relativistic(df: pl.DataFrame, **kwargs) -> go.Figure:
    """Quick relativistic effects visualization."""
    visualizer = SpaceDataVisualizer()
    return visualizer.plot_relativistic_effects(df, **kwargs)


def quick_dashboard(df: pl.DataFrame, save_path: Optional[Path] = None) -> str:
    """Quick dashboard creation."""
    visualizer = SpaceDataVisualizer()
    return visualizer.create_dashboard(df, save_path)
