"""
Synthetic Pharmaceutical Inventory Data Generator
Generates realistic pharmaceutical inventory data for testing and demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class PharmaceuticalDataGenerator:
    """Generate synthetic pharmaceutical inventory data"""

    def __init__(self, num_skus=10000, start_date='2023-01-01', end_date='2024-12-31'):
        self.num_skus = num_skus
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.days = (self.end_date - self.start_date).days + 1

        # Pharmaceutical product categories
        self.categories = {
            'Vaccines': 0.15,
            'Biologics': 0.25,
            'Small Molecules': 0.35,
            'Oncology': 0.15,
            'Specialty': 0.10
        }

        # Temperature requirements
        self.temp_requirements = [
            'Ultra-Cold (-80°C to -60°C)',
            'Frozen (-25°C to -15°C)',
            'Refrigerated (2°C to 8°C)',
            'Room Temperature (15°C to 25°C)',
            'Controlled Room Temperature (20°C to 25°C)'
        ]

        # Storage locations
        self.storage_locations = [
            'Warehouse-A-Zone-1', 'Warehouse-A-Zone-2', 'Warehouse-A-Zone-3',
            'Warehouse-B-Zone-1', 'Warehouse-B-Zone-2',
            'Distribution-Center-East', 'Distribution-Center-West',
            'Cold-Storage-Facility-1', 'Cold-Storage-Facility-2'
        ]

        # Manufacturing sites
        self.manufacturing_sites = [
            'Singapore-Plant-01', 'USA-NC-Plant-02', 'Ireland-Cork-Plant-03',
            'Germany-Munich-Plant-04', 'UK-London-Plant-05', 'Japan-Tokyo-Plant-06',
            'India-Hyderabad-Plant-07', 'Switzerland-Basel-Plant-08'
        ]

        # Suppliers
        self.suppliers = [
            'PharmaCorp International', 'BioTech Suppliers Ltd', 'MedSource Global',
            'LifeScience Partners', 'GlobalPharma Supply', 'BioPharma Solutions',
            'MedTech Distributors', 'Advanced Therapeutics Supply'
        ]

    def generate_sku_master(self):
        """Generate SKU master data"""
        print(f"Generating {self.num_skus} SKUs...")

        skus = []
        for i in range(self.num_skus):
            # Assign category
            category = np.random.choice(
                list(self.categories.keys()),
                p=list(self.categories.values())
            )

            # Generate SKU ID
            sku_id = f"SKU-{category[:3].upper()}-{i+1:06d}"

            # Generate product attributes based on category
            if category == 'Vaccines':
                dosage_forms = ['Injection', 'Nasal Spray']
                shelf_life_days = np.random.randint(180, 730)  # 6-24 months
                unit_cost = np.random.uniform(50, 500)
                criticality = np.random.choice(['High', 'Critical'], p=[0.3, 0.7])
                temp_req = np.random.choice(self.temp_requirements[:3])  # Cold storage
            elif category == 'Biologics':
                dosage_forms = ['Injection', 'Infusion', 'Subcutaneous']
                shelf_life_days = np.random.randint(365, 1095)  # 1-3 years
                unit_cost = np.random.uniform(100, 2000)
                criticality = np.random.choice(['High', 'Critical'], p=[0.5, 0.5])
                temp_req = np.random.choice(self.temp_requirements[1:4])  # Refrigerated/Frozen
            elif category == 'Oncology':
                dosage_forms = ['Tablet', 'Capsule', 'Injection', 'Infusion']
                shelf_life_days = np.random.randint(730, 1825)  # 2-5 years
                unit_cost = np.random.uniform(500, 5000)
                criticality = 'Critical'
                temp_req = np.random.choice(self.temp_requirements[2:])
            else:  # Small Molecules, Specialty
                dosage_forms = ['Tablet', 'Capsule', 'Syrup', 'Topical']
                shelf_life_days = np.random.randint(730, 1825)  # 2-5 years
                unit_cost = np.random.uniform(5, 200)
                criticality = np.random.choice(['Low', 'Medium', 'High'], p=[0.4, 0.4, 0.2])
                temp_req = np.random.choice(self.temp_requirements[3:])

            # Calculate ABC/XYZ classification
            # ABC based on value, XYZ based on demand variability
            abc_class = np.random.choice(['A', 'B', 'C'], p=[0.2, 0.3, 0.5])
            xyz_class = np.random.choice(['X', 'Y', 'Z'], p=[0.3, 0.4, 0.3])

            sku = {
                'sku_id': sku_id,
                'product_name': f"{category} Product {i+1}",
                'category': category,
                'dosage_form': np.random.choice(dosage_forms),
                'strength': f"{np.random.choice([5, 10, 25, 50, 100, 250, 500])}mg",
                'unit_cost': round(unit_cost, 2),
                'holding_cost_pct': round(np.random.uniform(0.20, 0.35), 3),  # 20-35% of unit cost annually
                'emergency_procurement_premium': round(np.random.uniform(1.5, 3.0), 2),  # 1.5x-3x normal cost
                'shelf_life_days': shelf_life_days,
                'temp_requirement': temp_req,
                'storage_location': np.random.choice(self.storage_locations),
                'manufacturing_site': np.random.choice(self.manufacturing_sites),
                'primary_supplier': np.random.choice(self.suppliers),
                'supplier_lead_time_days': np.random.randint(14, 90),
                'supplier_reliability_pct': round(np.random.uniform(0.85, 0.99), 3),
                'min_order_quantity': np.random.choice([100, 500, 1000, 5000]),
                'reorder_point': 0,  # Will be calculated
                'abc_classification': abc_class,
                'xyz_classification': xyz_class,
                'abc_xyz_class': f"{abc_class}{xyz_class}",
                'criticality': criticality,
                'is_controlled_substance': category == 'Oncology' or np.random.choice([True, False], p=[0.1, 0.9]),
                'requires_cold_chain': temp_req in self.temp_requirements[:3],
                'batch_tracked': True,
                'lot_tracked': True,
                'serialized': category in ['Vaccines', 'Biologics', 'Oncology']
            }

            skus.append(sku)

        df_skus = pd.DataFrame(skus)
        print(f"Generated {len(df_skus)} SKU records")
        return df_skus

    def generate_historical_consumption(self, df_skus):
        """Generate historical consumption data"""
        print(f"Generating {self.days} days of consumption data for {len(df_skus)} SKUs...")

        consumption_records = []

        for _, sku in df_skus.iterrows():
            # Base demand varies by category and ABC class
            if sku['abc_classification'] == 'A':
                base_demand = np.random.uniform(100, 500)
            elif sku['abc_classification'] == 'B':
                base_demand = np.random.uniform(20, 100)
            else:  # C
                base_demand = np.random.uniform(1, 20)

            # Demand variability based on XYZ class
            if sku['xyz_classification'] == 'X':
                cv = 0.1  # Low variability
            elif sku['xyz_classification'] == 'Y':
                cv = 0.3  # Medium variability
            else:  # Z
                cv = 0.6  # High variability

            # Generate time series with trend and seasonality
            date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')

            for date in date_range:
                # Trend component (slight growth over time)
                trend = 1 + (date - self.start_date).days / (365 * 5)  # 5-year growth factor

                # Seasonal component (quarterly patterns)
                day_of_year = date.timetuple().tm_yday
                seasonality = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)

                # Random component
                random_factor = np.random.normal(1, cv)

                # Calculate demand
                demand = base_demand * trend * seasonality * random_factor
                demand = max(0, int(demand))  # Ensure non-negative integer

                # Randomly introduce stockouts (5% probability for critical items)
                stockout = False
                if sku['criticality'] == 'Critical' and np.random.random() < 0.02:
                    stockout = True
                elif sku['criticality'] == 'High' and np.random.random() < 0.05:
                    stockout = True

                # Wastage due to expiration (higher for short shelf-life items)
                expiration_rate = 0.01 if sku['shelf_life_days'] < 365 else 0.001
                wastage = int(demand * expiration_rate * np.random.random())

                consumption_records.append({
                    'date': date,
                    'sku_id': sku['sku_id'],
                    'demand_units': demand,
                    'fulfilled_units': 0 if stockout else demand,
                    'stockout_flag': stockout,
                    'wastage_units': wastage,
                    'current_inventory': 0,  # Will be calculated in Silver layer
                    'replenishment_order': 0   # Will be calculated
                })

        df_consumption = pd.DataFrame(consumption_records)
        print(f"Generated {len(df_consumption):,} consumption records")
        return df_consumption

    def generate_supplier_performance(self, df_skus):
        """Generate supplier performance metrics"""
        print("Generating supplier performance data...")

        suppliers = df_skus['primary_supplier'].unique()
        performance_records = []

        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='M')

        for supplier in suppliers:
            for date in date_range:
                # On-time delivery varies by supplier
                base_otd = np.random.uniform(0.85, 0.98)
                otd_rate = min(1.0, base_otd + np.random.normal(0, 0.05))

                # Quality acceptance varies
                qa_rate = np.random.uniform(0.95, 0.999)

                # Lead time variance
                avg_lead_time = df_skus[df_skus['primary_supplier'] == supplier]['supplier_lead_time_days'].mean()
                actual_lead_time = int(avg_lead_time * (1 + np.random.normal(0, 0.2)))

                performance_records.append({
                    'month': date,
                    'supplier_name': supplier,
                    'on_time_delivery_rate': round(otd_rate, 3),
                    'quality_acceptance_rate': round(qa_rate, 3),
                    'avg_lead_time_days': actual_lead_time,
                    'lead_time_variance_days': int(abs(np.random.normal(0, 5))),
                    'orders_placed': np.random.randint(10, 100),
                    'orders_delivered': int(np.random.randint(10, 100) * otd_rate),
                    'defect_rate': round(1 - qa_rate, 4),
                    'price_variance_pct': round(np.random.normal(0, 0.05), 3)
                })

        df_supplier_perf = pd.DataFrame(performance_records)
        print(f"Generated {len(df_supplier_perf)} supplier performance records")
        return df_supplier_perf

    def generate_inventory_transactions(self, df_skus, df_consumption):
        """Generate inventory transaction log"""
        print("Generating inventory transactions...")

        # Sample transactions for demonstration
        transaction_records = []

        # Get sample of SKUs for transactions
        sample_skus = df_skus.sample(min(1000, len(df_skus)))

        for _, sku in sample_skus.iterrows():
            # Generate 50-100 transactions per SKU over the period
            num_transactions = np.random.randint(50, 100)

            transaction_dates = np.random.choice(
                pd.date_range(start=self.start_date, end=self.end_date, freq='D'),
                size=num_transactions,
                replace=True
            )

            for txn_date in sorted(transaction_dates):
                txn_type = np.random.choice(
                    ['RECEIPT', 'CONSUMPTION', 'ADJUSTMENT', 'TRANSFER', 'EXPIRATION'],
                    p=[0.3, 0.5, 0.05, 0.1, 0.05]
                )

                if txn_type == 'RECEIPT':
                    quantity = np.random.randint(100, 5000)
                    unit_cost = sku['unit_cost'] * np.random.uniform(0.95, 1.05)
                elif txn_type == 'CONSUMPTION':
                    quantity = -np.random.randint(10, 500)
                    unit_cost = sku['unit_cost']
                elif txn_type == 'EXPIRATION':
                    quantity = -np.random.randint(1, 100)
                    unit_cost = sku['unit_cost']
                else:  # ADJUSTMENT, TRANSFER
                    quantity = np.random.randint(-100, 100)
                    unit_cost = sku['unit_cost']

                transaction_records.append({
                    'transaction_id': f"TXN-{len(transaction_records)+1:010d}",
                    'transaction_date': txn_date,
                    'sku_id': sku['sku_id'],
                    'transaction_type': txn_type,
                    'quantity': quantity,
                    'unit_cost': round(unit_cost, 2),
                    'total_value': round(quantity * unit_cost, 2),
                    'lot_number': f"LOT-{np.random.randint(100000, 999999)}",
                    'expiration_date': txn_date + timedelta(days=sku['shelf_life_days']),
                    'storage_location': sku['storage_location'],
                    'reference_doc': f"PO-{np.random.randint(10000, 99999)}" if txn_type == 'RECEIPT' else None
                })

        df_transactions = pd.DataFrame(transaction_records)
        print(f"Generated {len(df_transactions):,} inventory transactions")
        return df_transactions

    def save_datasets(self, output_dir='data/bronze'):
        """Generate and save all datasets"""
        print("\n" + "="*80)
        print("PHARMACEUTICAL INVENTORY DATA GENERATION")
        print("="*80 + "\n")

        # Generate datasets
        df_skus = self.generate_sku_master()
        df_consumption = self.generate_historical_consumption(df_skus)
        df_supplier_perf = self.generate_supplier_performance(df_skus)
        df_transactions = self.generate_inventory_transactions(df_skus, df_consumption)

        # Save to parquet (efficient columnar format)
        print(f"\nSaving datasets to {output_dir}...")

        df_skus.to_parquet(f'{output_dir}/sku_master.parquet', index=False)
        df_consumption.to_parquet(f'{output_dir}/consumption_history.parquet', index=False)
        df_supplier_perf.to_parquet(f'{output_dir}/supplier_performance.parquet', index=False)
        df_transactions.to_parquet(f'{output_dir}/inventory_transactions.parquet', index=False)

        # Also save as CSV for inspection
        df_skus.to_csv(f'{output_dir}/sku_master.csv', index=False)
        df_consumption.head(10000).to_csv(f'{output_dir}/consumption_history_sample.csv', index=False)
        df_supplier_perf.to_csv(f'{output_dir}/supplier_performance.csv', index=False)
        df_transactions.head(10000).to_csv(f'{output_dir}/inventory_transactions_sample.csv', index=False)

        # Print summary statistics
        print("\n" + "="*80)
        print("DATA GENERATION SUMMARY")
        print("="*80)
        print(f"SKU Master Records: {len(df_skus):,}")
        print(f"Consumption Records: {len(df_consumption):,}")
        print(f"Supplier Performance Records: {len(df_supplier_perf):,}")
        print(f"Inventory Transactions: {len(df_transactions):,}")
        print(f"\nTotal Records Generated: {len(df_skus) + len(df_consumption) + len(df_supplier_perf) + len(df_transactions):,}")
        print(f"Time Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Number of Days: {self.days}")
        print("\nCategory Distribution:")
        print(df_skus['category'].value_counts())
        print("\nABC-XYZ Classification:")
        print(df_skus['abc_xyz_class'].value_counts().head(10))
        print("="*80 + "\n")

        return {
            'skus': df_skus,
            'consumption': df_consumption,
            'supplier_performance': df_supplier_perf,
            'transactions': df_transactions
        }


if __name__ == "__main__":
    # Initialize generator
    generator = PharmaceuticalDataGenerator(
        num_skus=10000,
        start_date='2023-01-01',
        end_date='2024-12-31'
    )

    # Generate and save datasets
    datasets = generator.save_datasets(output_dir='data/bronze')

    print("Data generation completed successfully!")
