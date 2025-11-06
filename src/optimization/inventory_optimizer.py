"""
Inventory Optimization Algorithms
Economic Order Quantity, Safety Stock, Multi-Echelon Optimization
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class InventoryOptimizer:
    """
    Comprehensive inventory optimization class
    Implements EOQ, Safety Stock, Reorder Point, and Multi-Echelon optimization
    """

    def __init__(self):
        self.z_score_map = {
            0.90: 1.28,
            0.95: 1.65,
            0.98: 2.05,
            0.99: 2.33,
            0.995: 2.58,
            0.999: 3.09
        }

    def calculate_eoq(
        self,
        annual_demand: float,
        ordering_cost: float,
        holding_cost_rate: float,
        unit_cost: float
    ) -> Dict[str, float]:
        """
        Calculate Economic Order Quantity (EOQ)

        Formula: EOQ = sqrt((2 * D * S) / H)
        Where:
        - D = Annual demand
        - S = Ordering cost per order
        - H = Annual holding cost per unit

        Args:
            annual_demand: Annual demand in units
            ordering_cost: Fixed cost per order
            holding_cost_rate: Holding cost as % of unit cost (e.g., 0.25 = 25%)
            unit_cost: Cost per unit

        Returns:
            Dictionary with EOQ and related metrics
        """
        # Calculate annual holding cost per unit
        annual_holding_cost = holding_cost_rate * unit_cost

        # EOQ formula
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / annual_holding_cost)

        # Number of orders per year
        num_orders = annual_demand / eoq

        # Total annual cost
        total_ordering_cost = num_orders * ordering_cost
        total_holding_cost = (eoq / 2) * annual_holding_cost
        total_cost = total_ordering_cost + total_holding_cost

        # Time between orders (in days)
        time_between_orders = 365 / num_orders

        return {
            'eoq': round(eoq, 2),
            'num_orders_per_year': round(num_orders, 2),
            'time_between_orders_days': round(time_between_orders, 2),
            'total_annual_cost': round(total_cost, 2),
            'annual_ordering_cost': round(total_ordering_cost, 2),
            'annual_holding_cost': round(total_holding_cost, 2)
        }

    def calculate_safety_stock(
        self,
        demand_std: float,
        lead_time_days: float,
        service_level: float = 0.98
    ) -> Dict[str, float]:
        """
        Calculate Safety Stock using demand variability

        Formula: SS = Z * sqrt(LT) * σ_demand
        Where:
        - Z = Z-score for desired service level
        - LT = Lead time
        - σ_demand = Standard deviation of demand

        Args:
            demand_std: Standard deviation of daily demand
            lead_time_days: Lead time in days
            service_level: Desired service level (e.g., 0.98 = 98%)

        Returns:
            Dictionary with safety stock and related metrics
        """
        # Get Z-score for service level
        z_score = self.z_score_map.get(service_level, stats.norm.ppf(service_level))

        # Safety stock formula
        safety_stock = z_score * np.sqrt(lead_time_days) * demand_std

        return {
            'safety_stock': round(safety_stock, 2),
            'service_level': service_level,
            'z_score': round(z_score, 2),
            'lead_time_days': lead_time_days
        }

    def calculate_reorder_point(
        self,
        average_daily_demand: float,
        lead_time_days: float,
        safety_stock: float
    ) -> float:
        """
        Calculate Reorder Point (ROP)

        Formula: ROP = (Average Daily Demand × Lead Time) + Safety Stock

        Args:
            average_daily_demand: Average demand per day
            lead_time_days: Lead time in days
            safety_stock: Safety stock units

        Returns:
            Reorder point in units
        """
        reorder_point = (average_daily_demand * lead_time_days) + safety_stock
        return round(reorder_point, 2)

    def calculate_service_level_from_stockout(
        self,
        stockout_probability: float
    ) -> float:
        """
        Calculate service level from stockout probability

        Args:
            stockout_probability: Probability of stockout (e.g., 0.02 = 2%)

        Returns:
            Service level (e.g., 0.98 = 98%)
        """
        return 1 - stockout_probability

    def optimize_inventory_policy(
        self,
        sku_id: str,
        annual_demand: float,
        demand_std: float,
        lead_time_days: float,
        unit_cost: float,
        ordering_cost: float = 100.0,
        holding_cost_rate: float = 0.25,
        service_level: float = 0.98
    ) -> Dict:
        """
        Comprehensive inventory policy optimization for a single SKU

        Args:
            sku_id: SKU identifier
            annual_demand: Annual demand in units
            demand_std: Standard deviation of daily demand
            lead_time_days: Lead time in days
            unit_cost: Cost per unit
            ordering_cost: Fixed ordering cost
            holding_cost_rate: Annual holding cost rate
            service_level: Target service level

        Returns:
            Complete inventory policy dictionary
        """
        # Calculate EOQ
        eoq_results = self.calculate_eoq(
            annual_demand,
            ordering_cost,
            holding_cost_rate,
            unit_cost
        )

        # Calculate safety stock
        safety_stock_results = self.calculate_safety_stock(
            demand_std,
            lead_time_days,
            service_level
        )

        # Calculate average daily demand
        avg_daily_demand = annual_demand / 365

        # Calculate reorder point
        reorder_point = self.calculate_reorder_point(
            avg_daily_demand,
            lead_time_days,
            safety_stock_results['safety_stock']
        )

        # Calculate maximum inventory
        max_inventory = reorder_point + eoq_results['eoq']

        # Calculate average inventory
        avg_inventory = safety_stock_results['safety_stock'] + (eoq_results['eoq'] / 2)

        # Calculate inventory turnover
        inventory_turnover = annual_demand / avg_inventory if avg_inventory > 0 else 0

        # Calculate days of supply
        days_of_supply = avg_inventory / avg_daily_demand if avg_daily_demand > 0 else 0

        return {
            'sku_id': sku_id,
            'optimal_order_quantity': eoq_results['eoq'],
            'reorder_point': reorder_point,
            'safety_stock': safety_stock_results['safety_stock'],
            'max_inventory': round(max_inventory, 2),
            'avg_inventory': round(avg_inventory, 2),
            'num_orders_per_year': eoq_results['num_orders_per_year'],
            'time_between_orders_days': eoq_results['time_between_orders_days'],
            'total_annual_cost': eoq_results['total_annual_cost'],
            'inventory_turnover': round(inventory_turnover, 2),
            'days_of_supply': round(days_of_supply, 2),
            'service_level': service_level
        }

    def calculate_abc_classification(
        self,
        df: pd.DataFrame,
        value_column: str = 'annual_value'
    ) -> pd.DataFrame:
        """
        Perform ABC classification on inventory

        A items: Top 20% by value (typically 80% of total value)
        B items: Next 30% by value (typically 15% of total value)
        C items: Bottom 50% by value (typically 5% of total value)

        Args:
            df: DataFrame with SKU data
            value_column: Column name for annual value

        Returns:
            DataFrame with ABC classification added
        """
        # Sort by value in descending order
        df_sorted = df.sort_values(by=value_column, ascending=False).copy()

        # Calculate cumulative sum and percentage
        df_sorted['cumulative_value'] = df_sorted[value_column].cumsum()
        total_value = df_sorted[value_column].sum()
        df_sorted['cumulative_pct'] = (df_sorted['cumulative_value'] / total_value) * 100

        # Classify
        df_sorted['abc_class'] = 'C'
        df_sorted.loc[df_sorted['cumulative_pct'] <= 80, 'abc_class'] = 'A'
        df_sorted.loc[
            (df_sorted['cumulative_pct'] > 80) & (df_sorted['cumulative_pct'] <= 95),
            'abc_class'
        ] = 'B'

        return df_sorted

    def calculate_xyz_classification(
        self,
        df: pd.DataFrame,
        demand_std_column: str = 'demand_std',
        demand_mean_column: str = 'demand_mean'
    ) -> pd.DataFrame:
        """
        Perform XYZ classification based on demand variability

        X items: Coefficient of Variation (CV) < 0.2 (Low variability)
        Y items: 0.2 <= CV < 0.5 (Medium variability)
        Z items: CV >= 0.5 (High variability)

        Args:
            df: DataFrame with SKU data
            demand_std_column: Column name for demand standard deviation
            demand_mean_column: Column name for mean demand

        Returns:
            DataFrame with XYZ classification added
        """
        df_copy = df.copy()

        # Calculate coefficient of variation
        df_copy['cv'] = df_copy[demand_std_column] / df_copy[demand_mean_column]
        df_copy['cv'] = df_copy['cv'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Classify
        df_copy['xyz_class'] = 'Z'
        df_copy.loc[df_copy['cv'] < 0.2, 'xyz_class'] = 'X'
        df_copy.loc[(df_copy['cv'] >= 0.2) & (df_copy['cv'] < 0.5), 'xyz_class'] = 'Y'

        return df_copy

    def optimize_multi_echelon(
        self,
        network_config: Dict[str, List[Dict]],
        total_inventory_target: float,
        service_level_target: float = 0.98
    ) -> Dict[str, Dict]:
        """
        Multi-echelon inventory optimization

        Distributes inventory across multiple echelons (Manufacturing, DC, Warehouse, Retail)
        to minimize total costs while meeting service level targets

        Args:
            network_config: Network configuration with echelons and their properties
            total_inventory_target: Total inventory to distribute
            service_level_target: Target service level

        Returns:
            Optimal inventory allocation by echelon
        """
        # Simplified multi-echelon optimization
        # In production, this would use linear programming (PuLP or CVXPY)

        echelons = network_config.get('echelons', [])
        num_echelons = len(echelons)

        if num_echelons == 0:
            return {}

        # Allocate inventory using square root rule approximation
        # (More sophisticated allocation would use optimization solvers)
        allocation = {}

        # Base allocation (simplified)
        for i, echelon in enumerate(echelons):
            echelon_name = echelon['name']
            demand_rate = echelon.get('demand_rate', 1.0)
            lead_time = echelon.get('lead_time_days', 7)

            # Simplified allocation based on demand and lead time
            weight = demand_rate * np.sqrt(lead_time)
            allocation[echelon_name] = weight

        # Normalize to total inventory target
        total_weight = sum(allocation.values())
        for echelon_name in allocation:
            allocation[echelon_name] = {
                'optimal_inventory': round(
                    (allocation[echelon_name] / total_weight) * total_inventory_target,
                    2
                ),
                'service_level': service_level_target
            }

        return allocation


def example_usage():
    """Example usage of the InventoryOptimizer class"""
    optimizer = InventoryOptimizer()

    # Example 1: Single SKU optimization
    print("="*80)
    print("EXAMPLE 1: Single SKU Inventory Policy Optimization")
    print("="*80)

    policy = optimizer.optimize_inventory_policy(
        sku_id='SKU-VAC-001234',
        annual_demand=50000,
        demand_std=25,
        lead_time_days=21,
        unit_cost=150.00,
        ordering_cost=100.00,
        holding_cost_rate=0.25,
        service_level=0.98
    )

    for key, value in policy.items():
        print(f"{key}: {value}")

    # Example 2: Multi-echelon optimization
    print("\n" + "="*80)
    print("EXAMPLE 2: Multi-Echelon Inventory Optimization")
    print("="*80)

    network_config = {
        'echelons': [
            {'name': 'Manufacturing', 'demand_rate': 1000, 'lead_time_days': 30},
            {'name': 'Regional DC', 'demand_rate': 800, 'lead_time_days': 14},
            {'name': 'Local Warehouse', 'demand_rate': 500, 'lead_time_days': 7},
            {'name': 'Retail', 'demand_rate': 200, 'lead_time_days': 2}
        ]
    }

    multi_echelon_allocation = optimizer.optimize_multi_echelon(
        network_config=network_config,
        total_inventory_target=100000,
        service_level_target=0.98
    )

    for echelon, allocation in multi_echelon_allocation.items():
        print(f"\n{echelon}:")
        for key, value in allocation.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    example_usage()
