AI-Driven Inventory Optimization Engine 🏭

Overview

This project addresses the critical supply chain challenge of balancing stockouts vs. overstocking under uncertain conditions. I engineered a Python-based simulation engine that uses Machine Learning to forecast demand and automatically optimize inventory parameters.

Key Features

🤖 Demand Forecasting: Utilizes Random Forest Regression (Scikit-Learn) to predict daily product demand based on seasonality and trend analysis.

📉 Risk Analysis: Automates the calculation of Forecast Error (RMSE) to quantify demand uncertainty.

🛡️ Dynamic Safety Stock: Implements the statistical Square Root Law to calculate optimal Safety Stock levels for a 95% Service Level target.

🔄 Automated Replenishment: Simulates a 30-day warehouse cycle with a dynamic Reorder Point (ROP) logic that triggers purchase orders automatically.

Tech Stack

Language: Python 3.14

Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib

Concepts: Inventory Management, Safety Stock, Reorder Point, EOQ, Lead Time Variability.

Results

Successfully simulated a scenario where the system maintained a 95% Service Level despite random lead-time delays.

Visualized stock depletion and replenishment cycles to identify potential bottlenecks.
