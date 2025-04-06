import argparse
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load data
demand_df = pd.read_csv("demand_forecasting.csv")
inventory_df = pd.read_csv("inventory_monitoring.csv")
pricing_df = pd.read_csv("pricing_optimization.csv")

# Define Agents
class DemandForecastingAgent:
    def forecast(self, product_id):
        df = demand_df[demand_df['Product ID'] == int(product_id)]
        df = df.groupby('Date')['Sales Quantity'].sum().reset_index()
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        model = Prophet(daily_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=5)
        forecast = model.predict(future)
        forecast_df = forecast[['ds', 'yhat']].tail(5)
        print(forecast_df)
        forecast[['ds', 'yhat']].tail(5).to_csv(f"forecast_{product_id}.csv", index=False)
        model.plot(forecast)
        plt.title(f"Forecast for Product {product_id}")
        plt.xlabel("Date")
        plt.ylabel("Predicted Demand")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"forecast_plot_{product_id}.png")
        plt.close()
        return forecast_df

class InventoryAgent:
    def get_inventory_status(self, product_id, store_id):
        record = inventory_df[(inventory_df['Product ID'] == int(product_id)) & (inventory_df['Store ID'] == int(store_id))]
        if record.empty:
            raise ValueError("Inventory data not found for given product and store.")
        return record[['Stock Levels', 'Reorder Point']].iloc[0]

class PricingAgent:
    def suggest_price(self, product_id, store_id):
        record = pricing_df[(pricing_df['Product ID'] == int(product_id)) & (pricing_df['Store ID'] == int(store_id))]
        if record.empty:
            print("⚠️ Pricing data not found for given product and store. Using default price ₹100.00")
            return 100.00
        base_price = record['Price'].iloc[0]
        elasticity = record['Elasticity Index'].iloc[0]
        adjusted_price = base_price * (1 - 0.01 * elasticity)
        return round(adjusted_price, 2)

# Main Execution Function
def run_simulation(product_id, store_id):
    print(f"🔍 Forecasting demand for Product ID: {product_id}")
    demand_agent = DemandForecastingAgent()
    forecast = demand_agent.forecast(product_id)

    print(f"📦 Checking inventory for Store ID: {store_id}")
    inventory_agent = InventoryAgent()
    inventory_status = inventory_agent.get_inventory_status(product_id, store_id)
    print("Inventory Status:")
    print(inventory_status)

    print(f"💸 Optimizing price...")
    pricing_agent = PricingAgent()
    suggested_price = pricing_agent.suggest_price(product_id, store_id)
    print(f"✅ Suggested Price: ₹{suggested_price}")

# CLI Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Retail Inventory Optimizer")
    parser.add_argument("--product_id", type=int, required=True, help="Product ID to analyze")
    parser.add_argument("--store_id", type=int, required=True, help="Store ID to analyze")
    args = parser.parse_args()
    run_simulation(args.product_id, args.store_id)
