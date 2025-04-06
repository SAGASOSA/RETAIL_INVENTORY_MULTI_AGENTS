import argparse
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
demand_df = pd.read_csv("demand_forecasting.csv")
inventory_df = pd.read_csv("inventory_monitoring.csv")
pricing_df = pd.read_csv("pricing_optimization.csv")
supply_chain_df = pd.read_csv("supply_chain_data.csv")

# Agent 1: Demand Forecaster (Using XGBoost)
class DemandForecaster:
    def forecast(self, product_id, store_id):
        # Filter and preprocess historical data for forecasting
        df = demand_df[(demand_df['Product ID'] == product_id) & (demand_df['Store ID'] == store_id)]
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        
        X = df[['DayOfWeek', 'Month', 'Year']]  # Features
        y = df['Sales Quantity']  # Target variable

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train XGBoost model
        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(X_train, y_train)

        # Predict future demand
        future_features = np.array([[i % 7, (i // 7) % 12 + 1, 2025] for i in range(1, 6)])  # Next 5 days
        forecast = model.predict(future_features)

        # Plot and save forecast
        forecast_df = pd.DataFrame({
            'ds': pd.date_range(start='2025-05-01', periods=5),
            'yhat': forecast
        })
        
        forecast_df.to_csv(f"forecast_{product_id}_{store_id}.csv", index=False)
        forecast_df.plot(x='ds', y='yhat', title=f"Demand Forecast for Product {product_id} at Store {store_id}")
        plt.xlabel("Date")
        plt.ylabel("Predicted Demand")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"forecast_plot_{product_id}_{store_id}.png")
        plt.close()

        return forecast_df

# Agent 2: Inventory Optimizer (Using EOQ, Safety Stock, and Reorder Point)
class InventoryOptimizer:
    def calculate_inventory(self, product_id, store_id):
        record = inventory_df[(inventory_df['Product ID'] == product_id) & (inventory_df['Store ID'] == store_id)]
        if record.empty:
            raise ValueError("Inventory data not found for given product and store.")
        
        current_stock = record['Stock Levels'].iloc[0]
        historical_demand = demand_df[(demand_df['Product ID'] == product_id) & (demand_df['Store ID'] == store_id)]['Sales Quantity'].mean()
        lead_time = record['Lead Time'].iloc[0]  # In days
        safety_stock = historical_demand * lead_time  # Simple model for safety stock

        # EOQ Formula: sqrt((2 * Demand * Ordering Cost) / Holding Cost)
        ordering_cost = 50  # Assumed cost for ordering
        holding_cost = 2  # Assumed holding cost per unit
        eoq = np.sqrt((2 * historical_demand * ordering_cost) / holding_cost)

        reorder_point = historical_demand * lead_time + safety_stock

        return {
            'current_stock': current_stock,
            'eoq': eoq,
            'reorder_point': reorder_point
        }

# Agent 3: Pricing & Promotion Strategist
class PricingAndPromotionStrategist:
    def suggest_price(self, product_id, store_id):
        record = pricing_df[(pricing_df['Product ID'] == product_id) & (pricing_df['Store ID'] == store_id)]
        if record.empty:
            print("⚠️ Pricing data not found for given product and store. Using default price ₹100.00")
            return 100.00

        base_price = record['Price'].iloc[0]
        elasticity = record['Elasticity Index'].iloc[0]

        # Adjust price based on elasticity (simple model)
        adjusted_price = base_price * (1 - 0.01 * elasticity)
        return round(adjusted_price, 2)

    def apply_promotion(self, product_id, store_id, current_price):
        # Check if product is slow-moving and apply discounts
        slow_moving_threshold = 100  # Example threshold for slow-moving inventory
        sales_data = demand_df[(demand_df['Product ID'] == product_id) & (demand_df['Store ID'] == store_id)]
        average_sales = sales_data['Sales Quantity'].mean()

        if average_sales < slow_moving_threshold:
            # Apply 10% discount on slow-moving products
            new_price = current_price * 0.90
            print(f"🎯 Applying promotion: New Price ₹{new_price}")
            return round(new_price, 2)
        return current_price

# Agent 4: Supply Chain Coordinator
class SupplyChainCoordinator:
    def optimize_inventory_flow(self, product_id, store_id):
        # Find surplus or understocked stores
        record = inventory_df[(inventory_df['Product ID'] == product_id) & (inventory_df['Store ID'] == store_id)]
        if record.empty:
            raise ValueError("Inventory data not found for given product and store.")
        
        current_stock = record['Stock Levels'].iloc[0]
        reorder_point = record['Reorder Point'].iloc[0]

        if current_stock > reorder_point:
            # Surplus stock, find stores that are understocked
            understocked_stores = inventory_df[(inventory_df['Product ID'] == product_id) & (inventory_df['Stock Levels'] < reorder_point)]
            for _, store in understocked_stores.iterrows():
                surplus = current_stock - reorder_point
                print(f"🔄 Routing surplus stock to Store ID {store['Store ID']} with {surplus} units")
                # Transfer stock to understocked stores
        else:
            print("⚠️ Inventory is understocked, triggering reorder process.")

# Main Execution Function
def run_simulation(product_id, store_id):
    print(f"🔍 Forecasting demand for Product ID: {product_id} at Store ID: {store_id}")
    demand_agent = DemandForecaster()
    demand_forecast = demand_agent.forecast(product_id, store_id)

    print(f"📦 Optimizing inventory for Product ID: {product_id} at Store ID: {store_id}")
    inventory_agent = InventoryOptimizer()
    inventory_status = inventory_agent.calculate_inventory(product_id, store_id)
    print(f"Inventory Status: {inventory_status}")

    print(f"💸 Suggesting optimal price for Product ID: {product_id} at Store ID: {store_id}")
    pricing_agent = PricingAndPromotionStrategist()
    suggested_price = pricing_agent.suggest_price(product_id, store_id)
    print(f"Suggested Price: ₹{suggested_price}")
    
    # Applying promotion if necessary
    final_price = pricing_agent.apply_promotion(product_id, store_id, suggested_price)
    print(f"Final Price after Promotion: ₹{final_price}")

    print(f"🔗 Optimizing inventory flow and supply chain for Product ID: {product_id}")
    supply_chain_agent = SupplyChainCoordinator()
    supply_chain_agent.optimize_inventory_flow(product_id, store_id)

# CLI Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Retail Inventory Optimizer")
    parser.add_argument("--product_id", type=int, required=True, help="Product ID to analyze")
    parser.add_argument("--store_id", type=int, required=True, help="Store ID to analyze")
    args = parser.parse_args()
    run_simulation(args.product_id, args.store_id)
