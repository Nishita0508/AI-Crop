"""
db.py
MongoDB connection setup using PyMongo.
Provides reusable collection objects for the whole app.
"""

from pymongo import MongoClient
from datetime import datetime

# ----------------------------
# MongoDB Connection
# ----------------------------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "AgriAI_DB"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# ----------------------------
# Collections
# ----------------------------
users_collection = db["users"]
crop_predictions_collection = db["crop_predictions"]
disease_predictions_collection = db["disease_predictions"]
disease_solutions_collection = db["disease_solutions"]


def init_indexes():
    """Create indexes for performance and uniqueness constraints."""
    users_collection.create_index("email", unique=True)
    users_collection.create_index("username", unique=True)
    crop_predictions_collection.create_index("user_id")
    disease_predictions_collection.create_index("user_id")
    disease_solutions_collection.create_index("disease_name", unique=True)


def get_server_time():
    return datetime.utcnow()
