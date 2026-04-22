from pymongo import MongoClient
import json
import os
from datetime import datetime

# ====================== CONFIG ======================
# TODO: Replace with your MongoDB Atlas connection string
MONGO_URI = "mongodb+srv://zainsheikh:<db_password>@cluster0.zszp3y1.mongodb.net/?appName=Cluster0"
DATABASE_NAME = "senticare"
COLLECTION_NAME = "cbt_templates"
# ===================================================

def seed_cbt_templates():
    try:
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        # Clear existing templates (safe for development)
        collection.delete_many({"emotion": "anxious"})

        # Load templates from JSON file
        json_path = os.path.join(os.path.dirname(__file__), "cbt_templates.json")
        with open(json_path, "r", encoding="utf-8") as f:
            templates = json.load(f)

        # Add timestamp
        for template in templates:
            template["created_at"] = datetime.utcnow()
            template["updated_at"] = datetime.utcnow()

        # Insert all templates
        result = collection.insert_many(templates)
        
        print(f"✅ Successfully seeded {len(result.inserted_ids)} CBT templates for anxiety!")
        print(f"Collection: {DATABASE_NAME}.{COLLECTION_NAME}")
        
    except Exception as e:
        print(f"❌ Error seeding templates: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    print("🚀 Starting CBT Templates Seed...")
    seed_cbt_templates()
