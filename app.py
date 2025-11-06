"""
Inventory Optimization & Replenishment Dashboard
Pharmaceutical Supply Chain Management Platform
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

server = app.server

# =============================================================================
# MOCK DATA GENERATION (For demonstration - Replace with actual data pipeline)
# =============================================================================

def generate_mock_kpi_data():
    """Generate mock KPI data"""
    return {
        'inventory_turnover': 11.5,
        'stockout_rate': 1.8,
        'carrying_cost': 2.3e6,
        'service_level': 98.2,
        'fill_rate': 97.8,
        'days_of_supply': 32,
        'total_skus': 10243,
        'critical_alerts': 47
    }

def generate_mock_consumption_trend():
    """Generate mock consumption trend data"""
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')

    np.random.seed(42)
    actual = 1000 + 200 * np.sin(np.linspace(0, 4*np.pi, 90)) + np.random.normal(0, 50, 90)
    forecast = 1000 + 200 * np.sin(np.linspace(0, 4*np.pi, 90)) + np.random.normal(0, 20, 90)

    return pd.DataFrame({
        'date': dates,
        'actual': actual.clip(min=0),
        'forecast': forecast.clip(min=0)
    })

def generate_mock_abc_xyz_data():
    """Generate ABC-XYZ classification matrix"""
    categories = ['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']
    values = [450, 520, 380, 890, 1100, 750, 1200, 1500, 3453]

    matrix = []
    for i, abc in enumerate(['A', 'B', 'C']):
        for j, xyz in enumerate(['X', 'Y', 'Z']):
            matrix.append({
                'ABC': abc,
                'XYZ': xyz,
                'Count': values[i*3 + j],
                'Category': abc + xyz
            })

    return pd.DataFrame(matrix)

def generate_mock_alerts():
    """Generate mock alert data"""
    alerts = []
    alert_types = [
        ('Stock Below Reorder Point', 'warning', 'Vaccine SKU-VAC-001234'),
        ('Expiring Within 30 Days', 'danger', 'Biologic SKU-BIO-005678'),
        ('Temperature Deviation', 'danger', 'Cold Storage Zone A'),
        ('High Demand Variance', 'info', 'Oncology SKU-ONC-009876'),
        ('Supplier Delay', 'warning', 'PharmaCorp International'),
    ]

    for i, (msg, severity, item) in enumerate(alert_types[:5]):
        alerts.append({
            'id': i + 1,
            'timestamp': datetime.now() - timedelta(hours=i),
            'severity': severity,
            'message': msg,
            'item': item
        })

    return alerts

def generate_mock_reorder_recommendations():
    """Generate reorder recommendations"""
    return pd.DataFrame({
        'sku_id': ['SKU-VAC-001234', 'SKU-BIO-005678', 'SKU-SMA-009876', 'SKU-ONC-003456'],
        'product_name': ['Vaccine Product A', 'Biologic Product B', 'Small Molecule C', 'Oncology Drug D'],
        'current_stock': [450, 1200, 8900, 340],
        'reorder_point': [500, 1500, 10000, 400],
        'optimal_order_qty': [1500, 3000, 15000, 800],
        'estimated_cost': [75000, 600000, 300000, 400000],
        'priority': ['High', 'Critical', 'Medium', 'High'],
        'lead_time_days': [21, 35, 14, 45]
    })

def generate_mock_supplier_performance():
    """Generate supplier performance data"""
    return pd.DataFrame({
        'supplier': ['PharmaCorp Intl', 'BioTech Suppliers', 'MedSource Global', 'GlobalPharma Supply'],
        'otd_rate': [0.95, 0.89, 0.97, 0.92],
        'quality_rate': [0.998, 0.995, 0.999, 0.996],
        'avg_lead_time': [28, 35, 21, 30],
        'orders_ytd': [245, 189, 312, 267]
    })

# =============================================================================
# UI COMPONENTS
# =============================================================================

def create_kpi_card(title, value, icon, color="primary"):
    """Create a KPI card component"""
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.I(className=f"fas {icon} fa-2x mb-2", style={'color': f'var(--bs-{color})'}),
                html.H3(value, className="mb-0"),
                html.P(title, className="text-muted mb-0")
            ])
        ]),
        className="text-center h-100"
    )

def create_navigation_bar():
    """Create top navigation bar"""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.NavbarBrand([
                            html.I(className="fas fa-boxes me-2"),
                            "Inventory Optimization & Replenishment"
                        ], className="d-flex align-items-center")
                    ])
                ])
            ], align="center", className="g-0 w-100")
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-4"
    )

def create_sidebar():
    """Create sidebar navigation"""
    return html.Div([
        html.H5("Navigation", className="mb-4"),
        dbc.Nav([
            dbc.NavLink([
                html.I(className="fas fa-chart-line me-2"),
                "Overview"
            ], href="/", active="exact", className="mb-2"),
            dbc.NavLink([
                html.I(className="fas fa-boxes me-2"),
                "Inventory Monitoring"
            ], href="/inventory", active="exact", className="mb-2"),
            dbc.NavLink([
                html.I(className="fas fa-brain me-2"),
                "Predictive Analytics"
            ], href="/predictive", active="exact", className="mb-2"),
            dbc.NavLink([
                html.I(className="fas fa-cogs me-2"),
                "Optimization"
            ], href="/optimization", active="exact", className="mb-2"),
            dbc.NavLink([
                html.I(className="fas fa-truck me-2"),
                "Supplier Performance"
            ], href="/suppliers", active="exact", className="mb-2"),
            dbc.NavLink([
                html.I(className="fas fa-bell me-2"),
                "Alerts & Notifications"
            ], href="/alerts", active="exact", className="mb-2"),
        ], vertical=True, pills=True)
    ], className="p-3 bg-light", style={"height": "100vh", "position": "fixed", "width": "250px"})

# =============================================================================
# PAGE LAYOUTS
# =============================================================================

def create_overview_page():
    """Create overview dashboard page"""
    kpi_data = generate_mock_kpi_data()
    consumption_df = generate_mock_consumption_trend()
    abc_xyz_df = generate_mock_abc_xyz_data()

    # Create consumption trend chart
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=consumption_df['date'],
        y=consumption_df['actual'],
        mode='lines',
        name='Actual',
        line=dict(color='#2E86AB', width=2)
    ))
    fig_trend.add_trace(go.Scatter(
        x=consumption_df['date'],
        y=consumption_df['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='#A23B72', width=2, dash='dash')
    ))
    fig_trend.update_layout(
        title="Consumption Trend - Last 90 Days",
        xaxis_title="Date",
        yaxis_title="Units",
        hovermode='x unified',
        template="plotly_white",
        height=350
    )

    # Create ABC-XYZ heatmap
    pivot_data = abc_xyz_df.pivot(index='ABC', columns='XYZ', values='Count')
    fig_heatmap = px.imshow(
        pivot_data,
        labels=dict(x="Demand Variability (XYZ)", y="Value Classification (ABC)", color="SKU Count"),
        x=['X (Low Var)', 'Y (Med Var)', 'Z (High Var)'],
        y=['A (High Value)', 'B (Med Value)', 'C (Low Value)'],
        color_continuous_scale='Blues',
        title="ABC-XYZ Classification Matrix"
    )
    fig_heatmap.update_layout(template="plotly_white", height=350)

    return html.Div([
        html.H2("Dashboard Overview", className="mb-4"),

        # KPI Cards
        dbc.Row([
            dbc.Col(create_kpi_card(
                "Inventory Turnover",
                f"{kpi_data['inventory_turnover']}x/year",
                "fa-sync-alt",
                "success"
            ), md=3, className="mb-4"),
            dbc.Col(create_kpi_card(
                "Stockout Rate",
                f"{kpi_data['stockout_rate']}%",
                "fa-exclamation-triangle",
                "warning"
            ), md=3, className="mb-4"),
            dbc.Col(create_kpi_card(
                "Carrying Cost",
                f"${kpi_data['carrying_cost']/1e6:.1f}M",
                "fa-dollar-sign",
                "info"
            ), md=3, className="mb-4"),
            dbc.Col(create_kpi_card(
                "Service Level",
                f"{kpi_data['service_level']}%",
                "fa-check-circle",
                "success"
            ), md=3, className="mb-4"),
        ]),

        dbc.Row([
            dbc.Col(create_kpi_card(
                "Fill Rate",
                f"{kpi_data['fill_rate']}%",
                "fa-boxes",
                "primary"
            ), md=3, className="mb-4"),
            dbc.Col(create_kpi_card(
                "Days of Supply",
                f"{kpi_data['days_of_supply']} days",
                "fa-calendar-alt",
                "info"
            ), md=3, className="mb-4"),
            dbc.Col(create_kpi_card(
                "Total SKUs",
                f"{kpi_data['total_skus']:,}",
                "fa-list",
                "secondary"
            ), md=3, className="mb-4"),
            dbc.Col(create_kpi_card(
                "Critical Alerts",
                f"{kpi_data['critical_alerts']}",
                "fa-bell",
                "danger"
            ), md=3, className="mb-4"),
        ]),

        # Charts
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    dcc.Graph(figure=fig_trend)
                ]))
            ], md=8, className="mb-4"),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    dcc.Graph(figure=fig_heatmap)
                ]))
            ], md=4, className="mb-4"),
        ])
    ])

def create_inventory_monitoring_page():
    """Create inventory monitoring page"""
    reorder_df = generate_mock_reorder_recommendations()

    return html.Div([
        html.H2("Real-Time Inventory Monitoring", className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Inventory Status by Category"),
                    dcc.Graph(
                        figure=px.bar(
                            pd.DataFrame({
                                'Category': ['Vaccines', 'Biologics', 'Small Molecules', 'Oncology', 'Specialty'],
                                'Stock Value ($M)': [12.5, 28.3, 45.2, 18.7, 9.8],
                                'Units (K)': [125, 85, 450, 62, 38]
                            }),
                            x='Category',
                            y='Stock Value ($M)',
                            title='',
                            color='Stock Value ($M)',
                            color_continuous_scale='Teal'
                        ).update_layout(template="plotly_white", height=300, showlegend=False)
                    )
                ]))
            ], md=6, className="mb-4"),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Temperature-Controlled Inventory"),
                    dcc.Graph(
                        figure=px.pie(
                            pd.DataFrame({
                                'Temp Requirement': ['Ultra-Cold', 'Frozen', 'Refrigerated', 'Room Temp'],
                                'Percentage': [8, 15, 35, 42]
                            }),
                            values='Percentage',
                            names='Temp Requirement',
                            title='',
                            color_discrete_sequence=px.colors.sequential.Blues_r
                        ).update_layout(template="plotly_white", height=300)
                    )
                ]))
            ], md=6, className="mb-4"),
        ]),

        # Reorder Recommendations Table
        dbc.Card(dbc.CardBody([
            html.H5("Immediate Reorder Recommendations", className="mb-3"),
            dbc.Table.from_dataframe(
                reorder_df,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                className="mb-0"
            )
        ]), className="mb-4")
    ])

def create_predictive_analytics_page():
    """Create predictive analytics page"""
    consumption_df = generate_mock_consumption_trend()

    # Extend forecast into future
    future_dates = pd.date_range(
        start=consumption_df['date'].max() + timedelta(days=1),
        periods=30,
        freq='D'
    )
    future_forecast = 1000 + 200 * np.sin(np.linspace(4*np.pi, 5*np.pi, 30)) + np.random.normal(0, 20, 30)

    fig_forecast = go.Figure()

    # Historical actual
    fig_forecast.add_trace(go.Scatter(
        x=consumption_df['date'][-30:],
        y=consumption_df['actual'][-30:],
        mode='lines',
        name='Historical',
        line=dict(color='#2E86AB', width=2)
    ))

    # Forecast
    fig_forecast.add_trace(go.Scatter(
        x=future_dates,
        y=future_forecast,
        mode='lines',
        name='30-Day Forecast',
        line=dict(color='#A23B72', width=2, dash='dash')
    ))

    # Confidence interval
    upper_bound = future_forecast + 100
    lower_bound = future_forecast - 100
    fig_forecast.add_trace(go.Scatter(
        x=future_dates.tolist() + future_dates.tolist()[::-1],
        y=upper_bound.tolist() + lower_bound.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(162, 59, 114, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        showlegend=True
    ))

    fig_forecast.update_layout(
        title="Demand Forecast - Next 30 Days",
        xaxis_title="Date",
        yaxis_title="Forecasted Units",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )

    return html.Div([
        html.H2("Predictive Analytics", className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Demand Forecasting Model", className="mb-3"),
                    html.P("Prophet Time Series Model - MAPE: 8.2%", className="text-muted"),
                    dcc.Graph(figure=fig_forecast)
                ]))
            ], md=12, className="mb-4")
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Lead Time Prediction"),
                    html.P("Average predicted lead time: 28.5 days", className="mb-2"),
                    html.P("Variance: ±4.2 days", className="mb-0 text-muted")
                ]))
            ], md=4, className="mb-4"),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Stockout Risk Analysis"),
                    html.P("High risk SKUs: 23", className="mb-2"),
                    html.P("Medium risk SKUs: 67", className="mb-0 text-muted")
                ]))
            ], md=4, className="mb-4"),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Forecast Accuracy"),
                    html.P("Last 30 days MAPE: 8.2%", className="mb-2"),
                    html.P("Trend: Improving (+1.3%)", className="mb-0 text-success")
                ]))
            ], md=4, className="mb-4"),
        ])
    ])

def create_optimization_page():
    """Create optimization page"""
    return html.Div([
        html.H2("Inventory Optimization", className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Economic Order Quantity (EOQ) Analysis"),
                    html.P("Optimized order quantities to minimize total costs", className="text-muted mb-3"),
                    html.Div([
                        html.H6("Top Savings Opportunities:"),
                        html.Ul([
                            html.Li("SKU-VAC-001234: $12,500 annual savings"),
                            html.Li("SKU-BIO-005678: $28,300 annual savings"),
                            html.Li("SKU-SMA-009876: $8,900 annual savings"),
                        ])
                    ])
                ]))
            ], md=6, className="mb-4"),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Safety Stock Optimization"),
                    html.P("Dynamic safety stock based on demand variability", className="text-muted mb-3"),
                    html.Div([
                        html.H6("Formula:"),
                        html.P("SS = Z × √(LT × σ²)", className="font-monospace text-primary"),
                        html.P("Current service level target: 98%", className="mb-0")
                    ])
                ]))
            ], md=6, className="mb-4"),
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Multi-Echelon Optimization"),
                    dcc.Graph(
                        figure=px.bar(
                            pd.DataFrame({
                                'Location': ['Manufacturing', 'Regional DC', 'Local Warehouse', 'Retail'],
                                'Optimal Stock (K Units)': [450, 280, 150, 80],
                                'Current Stock (K Units)': [500, 250, 180, 70]
                            }),
                            x='Location',
                            y=['Optimal Stock (K Units)', 'Current Stock (K Units)'],
                            barmode='group',
                            title='Optimal vs Current Stock Levels by Location'
                        ).update_layout(template="plotly_white", height=350)
                    )
                ]))
            ], md=12, className="mb-4")
        ])
    ])

def create_supplier_performance_page():
    """Create supplier performance page"""
    supplier_df = generate_mock_supplier_performance()

    fig_otd = px.bar(
        supplier_df,
        x='supplier',
        y='otd_rate',
        title='On-Time Delivery Rate by Supplier',
        labels={'otd_rate': 'OTD Rate (%)', 'supplier': 'Supplier'},
        color='otd_rate',
        color_continuous_scale='Greens'
    )
    fig_otd.update_layout(template="plotly_white", height=300, showlegend=False)

    return html.Div([
        html.H2("Supplier Performance", className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Supplier Performance Metrics"),
                    dcc.Graph(figure=fig_otd)
                ]))
            ], md=12, className="mb-4")
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Supplier Scorecard"),
                    dbc.Table.from_dataframe(
                        supplier_df,
                        striped=True,
                        bordered=True,
                        hover=True,
                        responsive=True,
                        className="mb-0"
                    )
                ]))
            ], md=12, className="mb-4")
        ])
    ])

def create_alerts_page():
    """Create alerts and notifications page"""
    alerts = generate_mock_alerts()

    alert_cards = []
    for alert in alerts:
        color_map = {'danger': 'danger', 'warning': 'warning', 'info': 'info'}
        alert_cards.append(
            dbc.Alert([
                html.H5([
                    html.I(className=f"fas fa-exclamation-circle me-2"),
                    alert['message']
                ], className="alert-heading"),
                html.P(f"Item: {alert['item']}", className="mb-1"),
                html.Small(f"Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}", className="text-muted")
            ], color=color_map.get(alert['severity'], 'secondary'), className="mb-3")
        )

    return html.Div([
        html.H2("Alerts & Notifications", className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H5("Active Alerts", className="mb-3"),
                html.Div(alert_cards)
            ], md=12)
        ])
    ])

# =============================================================================
# MAIN LAYOUT
# =============================================================================

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    create_navigation_bar(),
    dbc.Container([
        dbc.Row([
            dbc.Col(create_sidebar(), width=2, className="p-0"),
            dbc.Col([
                html.Div(id='page-content', className="p-4")
            ], width=10)
        ])
    ], fluid=True)
])

# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    """Route to different pages based on URL"""
    if pathname == '/inventory':
        return create_inventory_monitoring_page()
    elif pathname == '/predictive':
        return create_predictive_analytics_page()
    elif pathname == '/optimization':
        return create_optimization_page()
    elif pathname == '/suppliers':
        return create_supplier_performance_page()
    elif pathname == '/alerts':
        return create_alerts_page()
    else:  # Default to overview
        return create_overview_page()

# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('DEBUG', 'True') == 'True'
    app.run_server(debug=debug, host='0.0.0.0', port=port)
