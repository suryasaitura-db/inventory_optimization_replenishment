"""
FastAPI REST API for Inventory Optimization System
ERP Integration Endpoints
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, date
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src.optimization.inventory_optimizer import InventoryOptimizer
except ImportError:
    print("Warning: Could not import InventoryOptimizer")

# Initialize FastAPI app
app = FastAPI(
    title="Inventory Optimization & Replenishment API",
    description="REST API for pharmaceutical inventory optimization and ERP integration",
    version="1.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize optimizer
optimizer = InventoryOptimizer()

# Pydantic models for request/response
class SKURequest(BaseModel):
    sku_id: str
    annual_demand: float
    demand_std: float
    lead_time_days: float
    unit_cost: float
    ordering_cost: float = 100.0
    holding_cost_rate: float = 0.25
    service_level: float = 0.98

class EOQRequest(BaseModel):
    annual_demand: float
    ordering_cost: float
    holding_cost_rate: float
    unit_cost: float

class SafetyStockRequest(BaseModel):
    demand_std: float
    lead_time_days: float
    service_level: float = 0.98

class InventoryStatus(BaseModel):
    sku_id: str
    current_stock: int
    reorder_point: float
    status: str  # "OK", "LOW", "CRITICAL"

# API Endpoints

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Inventory Optimization & Replenishment API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "inventory-optimization-api"
    }

@app.post("/api/v1/optimization/calculate-eoq")
def calculate_eoq(request: EOQRequest):
    """
    Calculate Economic Order Quantity

    Args:
        request: EOQ calculation parameters

    Returns:
        EOQ results including optimal order quantity and costs
    """
    try:
        result = optimizer.calculate_eoq(
            annual_demand=request.annual_demand,
            ordering_cost=request.ordering_cost,
            holding_cost_rate=request.holding_cost_rate,
            unit_cost=request.unit_cost
        )
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/optimization/safety-stock")
def calculate_safety_stock(request: SafetyStockRequest):
    """
    Calculate Safety Stock

    Args:
        request: Safety stock calculation parameters

    Returns:
        Safety stock level and related metrics
    """
    try:
        result = optimizer.calculate_safety_stock(
            demand_std=request.demand_std,
            lead_time_days=request.lead_time_days,
            service_level=request.service_level
        )
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/optimization/inventory-policy")
def optimize_inventory_policy(request: SKURequest):
    """
    Optimize complete inventory policy for a SKU

    Args:
        request: SKU and inventory parameters

    Returns:
        Complete inventory policy including EOQ, ROP, safety stock
    """
    try:
        result = optimizer.optimize_inventory_policy(
            sku_id=request.sku_id,
            annual_demand=request.annual_demand,
            demand_std=request.demand_std,
            lead_time_days=request.lead_time_days,
            unit_cost=request.unit_cost,
            ordering_cost=request.ordering_cost,
            holding_cost_rate=request.holding_cost_rate,
            service_level=request.service_level
        )
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/inventory/sku/{sku_id}")
def get_sku_details(sku_id: str):
    """
    Get SKU details (Mock implementation)

    Args:
        sku_id: SKU identifier

    Returns:
        SKU details including current stock and status
    """
    # Mock implementation - replace with actual database query
    return {
        "success": True,
        "data": {
            "sku_id": sku_id,
            "product_name": f"Product {sku_id}",
            "category": "Vaccines",
            "current_stock": 1250,
            "reorder_point": 500,
            "unit_cost": 150.00,
            "status": "OK"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/inventory/stock-levels")
def get_stock_levels(
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by status (OK/LOW/CRITICAL)")
):
    """
    Get current stock levels for all SKUs (Mock implementation)

    Args:
        category: Optional category filter
        status: Optional status filter

    Returns:
        List of SKUs with current stock levels
    """
    # Mock implementation - replace with actual database query
    mock_data = [
        {"sku_id": "SKU-VAC-001234", "current_stock": 450, "reorder_point": 500, "status": "LOW"},
        {"sku_id": "SKU-BIO-005678", "current_stock": 1200, "reorder_point": 1500, "status": "LOW"},
        {"sku_id": "SKU-SMA-009876", "current_stock": 8900, "reorder_point": 10000, "status": "LOW"},
        {"sku_id": "SKU-ONC-003456", "current_stock": 1500, "reorder_point": 400, "status": "OK"}
    ]

    # Apply filters
    filtered_data = mock_data
    if category:
        # Mock filtering - in production, query database with category filter
        filtered_data = [item for item in filtered_data if category.upper() in item['sku_id']]
    if status:
        filtered_data = [item for item in filtered_data if item['status'] == status.upper()]

    return {
        "success": True,
        "data": filtered_data,
        "count": len(filtered_data),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/reorder/recommendations")
def get_reorder_recommendations(min_priority: Optional[str] = "MEDIUM"):
    """
    Get reorder recommendations (Mock implementation)

    Args:
        min_priority: Minimum priority level (LOW/MEDIUM/HIGH/CRITICAL)

    Returns:
        List of recommended reorders
    """
    # Mock implementation
    recommendations = [
        {
            "sku_id": "SKU-VAC-001234",
            "product_name": "Vaccine Product A",
            "current_stock": 450,
            "reorder_point": 500,
            "optimal_order_qty": 1500,
            "estimated_cost": 75000,
            "priority": "HIGH",
            "lead_time_days": 21
        },
        {
            "sku_id": "SKU-BIO-005678",
            "product_name": "Biologic Product B",
            "current_stock": 1200,
            "reorder_point": 1500,
            "optimal_order_qty": 3000,
            "estimated_cost": 600000,
            "priority": "CRITICAL",
            "lead_time_days": 35
        }
    ]

    return {
        "success": True,
        "data": recommendations,
        "count": len(recommendations),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/suppliers/performance")
def get_supplier_performance():
    """
    Get supplier performance metrics (Mock implementation)

    Returns:
        List of suppliers with performance metrics
    """
    # Mock implementation
    suppliers = [
        {
            "supplier_name": "PharmaCorp International",
            "otd_rate": 0.95,
            "quality_rate": 0.998,
            "avg_lead_time": 28,
            "orders_ytd": 245
        },
        {
            "supplier_name": "BioTech Suppliers Ltd",
            "supplier_name": 0.89,
            "quality_rate": 0.995,
            "avg_lead_time": 35,
            "orders_ytd": 189
        }
    ]

    return {
        "success": True,
        "data": suppliers,
        "count": len(suppliers),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/forecast/demand/{sku_id}")
def get_demand_forecast(
    sku_id: str,
    periods: int = Query(30, ge=1, le=365, description="Number of periods to forecast")
):
    """
    Get demand forecast for SKU (Mock implementation)

    Args:
        sku_id: SKU identifier
        periods: Number of periods to forecast

    Returns:
        Demand forecast data
    """
    # Mock implementation
    import numpy as np
    dates = [(datetime.now() + timedelta(days=i)).date() for i in range(periods)]
    base_demand = 1000
    forecast_values = [base_demand + 100 * np.sin(i / 10) + np.random.normal(0, 20) for i in range(periods)]

    forecast_data = [
        {
            "date": str(dates[i]),
            "forecasted_demand": max(0, round(forecast_values[i], 2)),
            "lower_bound": max(0, round(forecast_values[i] - 100, 2)),
            "upper_bound": round(forecast_values[i] + 100, 2)
        }
        for i in range(periods)
    ]

    return {
        "success": True,
        "data": {
            "sku_id": sku_id,
            "forecast": forecast_data,
            "model": "Prophet",
            "mape": 8.2
        },
        "timestamp": datetime.now().isoformat()
    }


# Run server (for development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
