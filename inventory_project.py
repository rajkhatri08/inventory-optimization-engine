# ======================================================
# PROJECT: AI Supply Chain Optimization Engine
# ======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import timedelta, date

print("----------------------------------------------------------------")
print("🏭  STARTING AI INVENTORY OPTIMIZER ")
print("----------------------------------------------------------------")

# 1. SETTING RULES
TARGET_SERVICE_LEVEL = 0.95 
LEAD_TIME_AVG = 3           
LEAD_TIME_STD = 1           
CURRENT_STOCK = 200         

print(f"✅ Step 1: Rules Set. Target Service Level: {TARGET_SERVICE_LEVEL*100}%")

# 2. GENERATING MOCK DATA
print("\n🔄 Step 2: Generating 365 days of historical factory data...")
dates = pd.date_range(end=date.today(), periods=365)
data = []
for d in dates:
    base = 50
    weekend = 30 if d.weekday() >= 5 else 0
    season = 20 * np.sin(d.month / 12 * 6.28)
    noise = np.random.normal(0, 5)
    demand = int(base + weekend + season + noise)
    data.append([d, max(0, demand)])

df = pd.DataFrame(data, columns=['Date', 'Demand'])
print(f"   -> Generated {len(df)} records of sales history.")

# 3. TRAINING AI
print("\n🧠 Step 3: Training Random Forest AI...")
df['DayOfWeek'] = df['Date'].dt.weekday
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
X = df[['DayOfWeek', 'Month', 'Day']]
y = df['Demand']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
print("   -> Training Complete.")

# 4. SAFETY STOCK MATH
print("\n📐 Step 4: Optimizing Inventory Parameters...")
predictions = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, predictions))
print(f"   -> Model Forecast Error: +/- {rmse:.2f} units")

z_score = 1.645 
avg_demand = df['Demand'].mean()
safety_stock = int(z_score * np.sqrt((LEAD_TIME_AVG * (rmse**2)) + (avg_demand**2 * LEAD_TIME_STD**2)))
reorder_point = int((avg_demand * LEAD_TIME_AVG) + safety_stock)

print(f"   -> 🛡️  OPTIMAL SAFETY STOCK : {safety_stock} Units")
print(f"   -> 🔄  REORDER POINT        : {reorder_point} Units")

# 5. SIMULATION
print("\n🚀 Step 5: Running 30-Day Warehouse Simulation...")
future_dates = [dates[-1] + timedelta(days=x) for x in range(1, 31)]
inventory = CURRENT_STOCK
simulation_data = []
pending_orders = [] 

for d in future_dates:
    pred_demand = int(model.predict([[d.weekday(), d.month, d.day]])[0])
    
    # Receive
    arrived = 0
    new_pending = []
    for delivery_date, qty in pending_orders:
        if delivery_date <= d: arrived += qty
        else: new_pending.append((delivery_date, qty))
    pending_orders = new_pending
    inventory += arrived
    
    # Sell
    sales = min(inventory, pred_demand)
    inventory -= sales
    
    # Order Logic
    order_placed = 0
    if inventory < reorder_point:
        order_qty = 300 
        lead = int(np.random.normal(LEAD_TIME_AVG, LEAD_TIME_STD))
        arrival = d + timedelta(days=max(1, lead))
        pending_orders.append((arrival, order_qty))
        order_placed = order_qty
        print(f"   [ALERT] {d.strftime('%Y-%m-%d')}: Stock Low ({inventory}). Ordered {order_qty}.")
    
    simulation_data.append([d, inventory, order_placed])

# 6. SAVE GRAPH
print("\n📊 Step 6: Saving Performance Graph...")
sim_df = pd.DataFrame(simulation_data, columns=['Date', 'Inventory', 'Order_Placed'])
plt.figure(figsize=(14, 7)) # Make the chart wider
plt.plot(sim_df['Date'], sim_df['Inventory'], label='Inventory Level', color='blue', linewidth=2)
plt.axhline(safety_stock, color='red', linestyle='--', label=f'Safety Stock ({safety_stock})')
plt.axhline(reorder_point, color='orange', linestyle='--', label=f'Reorder Point ({reorder_point})')
orders = sim_df[sim_df['Order_Placed'] > 0]
if not orders.empty:
    plt.scatter(orders['Date'], orders['Inventory'], color='green', s=100, zorder=5, label='Order Triggered')

# --- THE FIX IS HERE ---
plt.xticks(rotation=45) # Rotate dates so they don't overlap!
plt.tight_layout()      # Make sure nothing gets cut off
# -----------------------

plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Dynamic Inventory Optimization: 30-Day Simulation")
plt.savefig('inventory_simulation_result.png')
print("✅ SUCCESS! Fixed Graph saved as 'inventory_simulation_result.png' on Desktop.")
