from pymongo import MongoClient
import pandas as pd
import json
import os
from src.constant import MONGO_DB_URL, MONGO_DATABASE_NAME, MONGO_COLLECTION_NAME

# Create a MongoDB client
client = MongoClient(MONGO_DB_URL)

# Dataset path
DATASET_FILE_PATH = os.path.join(os.getcwd(), "notebooks", "datasets", "UCI_Credit_Card.csv")

# Load dataset
df = pd.read_csv(DATASET_FILE_PATH)

# Convert DataFrame to JSON records
json_records = list(json.loads(df.T.to_json()).values())

# Insert JSON records into MongoDB collection
client[MONGO_DATABASE_NAME][MONGO_COLLECTION_NAME].insert_many(json_records)