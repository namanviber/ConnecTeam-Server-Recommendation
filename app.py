import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from lightfm import LightFM
from lightfm.data import Dataset

# Initialize Flask app
app = Flask(__name__)

# Paths for data and model files
DATA_PATH = "./Dataset"
MODEL_PATH = "./Model/lightfm_model.pkl"

# Load initial data and mappings
df_user = pd.read_csv(os.path.join(DATA_PATH, "user_data.csv"))
df_server = pd.read_csv(os.path.join(DATA_PATH, "server_data.csv"))
df_interaction = pd.read_csv(os.path.join(DATA_PATH, "server_interaction.csv"))

# Load category mapping
df_category = pd.read_csv(os.path.join(DATA_PATH, "server_category.csv"))
category_mapping = set(df_category['Category ID'])

# Mapping for users and servers
user_map = df_user.set_index('user_id')['user_num'].to_dict()
server_map = df_server.set_index('server_id')['server_num'].to_dict()

# Load the pre-trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Function to update datasets and mappings
def update_mappings(user_id, server_id):
    global df_user, df_server, user_map, server_map

    # Check and update user mapping
    if user_id not in user_map:
        new_user_num = len(user_map) + 1
        user_map[user_id] = new_user_num
        new_row = pd.DataFrame([{"user_id": user_id, "user_num": new_user_num}])
        df_user = pd.concat([df_user, new_row], ignore_index=True)
        df_user.to_csv(os.path.join(DATA_PATH, "user_data.csv"), index=False)

    # Check and update server mapping
    if server_id not in server_map:
        new_server_num = len(server_map) + 1
        server_map[server_id] = new_server_num
        new_row = pd.DataFrame([{"server_id": server_id, "server_num": new_server_num}])
        df_server = pd.concat([df_server, new_row], ignore_index=True)
        df_server.to_csv(os.path.join(DATA_PATH, "server_data.csv"), index=False)

# Function to retrain the model
def retrain_model():
    global model, df_interaction

    # Prepare interaction data
    df_interaction['user_num'] = df_interaction['user_id'].map(user_map)
    df_interaction['server_num'] = df_interaction['server_id'].map(server_map)
    interactions = [
        (row['user_num'], row['server_num'], row['rating']) for _, row in df_interaction.iterrows()
    ]

    # Prepare LightFM Dataset
    dataset = Dataset()
    dataset.fit(users=user_map.values(), items=server_map.values(), user_features=category_mapping, item_features=category_mapping)
    interactions_matrix, weights = dataset.build_interactions(interactions)

    # Build feature matrices
    user_features = dataset.build_user_features([(user_map[user], []) for user in user_map])
    server_features = dataset.build_item_features([(server_map[server], []) for server in server_map])

    # Retrain the model
    model = LightFM(no_components=150, learning_rate=0.05, loss='warp')
    model.fit(interactions_matrix, item_features=server_features, user_features=user_features, sample_weight=weights, epochs=10, num_threads=4)

    # Save the updated model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

# API endpoint to handle new interaction data
@app.route('/retrain', methods=['POST'])
def retrain():
    global df_interaction

    # Parse incoming data
    data = request.json  # Expected format: List of {user_id, server_id, rating, server_category}
    new_interactions = []

    for entry in data:
        user_id = entry['user_id']
        server_id = entry['server_id']
        rating = entry['rating']

        # Update mappings
        update_mappings(user_id, server_id)

        # Add to interactions
        new_interactions.append({
            "user_id": user_id,
            "server_id": server_id,
            "rating": rating
        })

    # Update interaction dataset
    new_df = pd.DataFrame(new_interactions)
    df_interaction = pd.concat([df_interaction, new_df], ignore_index=True)
    df_interaction.to_csv(os.path.join(DATA_PATH, "server_interaction.csv"), index=False)

    # Retrain the model
    retrain_model()

    return jsonify({"message": "Model retrained successfully!"})

# API endpoint for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json.get("user_id")

    # Check if the user exists
    if user_id not in user_map:
        return jsonify({"error": "User not found!"}), 404

    user_num = user_map[user_id]
    all_server_scores = model.predict(
        user_ids=user_num,
        item_ids=np.arange(len(server_map)),
    )

    # Get top recommendations
    top_servers = np.argsort(-all_server_scores)[:10]
    recommended_servers = df_server[df_server['server_num'].isin(top_servers)]['server_id'].tolist()

    return jsonify({"recommended_servers": recommended_servers})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5100)
