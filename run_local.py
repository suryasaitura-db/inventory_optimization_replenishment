#!/usr/bin/env python3
"""
Local development server for Inventory Optimization Dashboard
Run this script to start the application locally
"""

import os
import sys

# Set environment variables for local development
os.environ['DEBUG'] = 'True'
os.environ['ENV'] = 'development'
os.environ['PORT'] = '8050'

print("=" * 80)
print("INVENTORY OPTIMIZATION & REPLENISHMENT DASHBOARD")
print("=" * 80)
print("\nStarting local development server...")
print("Environment: DEVELOPMENT")
print("Debug Mode: ON")
print("\nServer will be available at:")
print("  -> http://localhost:8050")
print("  -> http://127.0.0.1:8050")
print("\nPress CTRL+C to stop the server")
print("=" * 80 + "\n")

# Import and run the app
try:
    from app import app
    app.run_server(
        debug=True,
        host='0.0.0.0',
        port=8050,
        dev_tools_hot_reload=True
    )
except ImportError as e:
    print(f"\n❌ ERROR: Missing dependencies. Please install requirements:")
    print("  pip install -r requirements.txt")
    print(f"\nDetails: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    sys.exit(1)
