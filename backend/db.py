import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)

db = client["senticare"]

users_collection = db["users"]
conversations_collection = db["conversations"]
cbt_collection = db["cbt_templates"]