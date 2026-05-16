# test_mongo.py
from db import client

try:
    client.admin.command("ping")
    print("✅ MongoDB connected successfully!")
except Exception as e:
    print(f"❌ Connection failed: {e}")