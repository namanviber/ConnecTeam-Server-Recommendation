import os
import json
import pandas as pd
from flask import Flask, request, jsonify
from recommendation_model import RecommendationSystem
from config import Config
import ast

app = Flask(__name__)
recommender = RecommendationSystem()

IS_MODEL_TRAINED = False

@app.route('/train', methods=['POST'])
def train_model():
    global IS_MODEL_TRAINED
    try:
        IS_MODEL_TRAINED = False
        data = request.json     
        df_interaction = pd.DataFrame(data)
        
        df_server = (
    df_interaction.groupby(['server_id', 'server_name'])
    .apply(lambda group: pd.Series({
        'server_category': sorted(set(item for sublist in group['server_category'].apply(ast.literal_eval) for item in sublist)),
        'server_image': group['server_image'].iloc[0], 
        'server_invite': group['server_invite'].iloc[0],
    }))
    .reset_index()
)
        df_user = df_interaction.groupby(['user_id', 'user_name'])['server_category'].apply(lambda x: sorted(set(item for sublist in x.apply(ast.literal_eval) for item in sublist))).reset_index()
        
        df_category = pd.read_csv("Dataset\server_category.csv")
        
        df_user['user_num'] = range(len(df_user))
        df_server['server_num'] = range(len(df_server))
        
        if recommender.prepare_data(df_interaction, df_server, df_user, df_category):
            if recommender.train_model():
                recommender.save_model(Config.MODEL_PATH)
                IS_MODEL_TRAINED = True
                return jsonify({"status": "Model Training Complete"}), 200
        
        return jsonify({"status": "Model Training Failed"}), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_id = request.json.get('user_id')
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400
        
        if not IS_MODEL_TRAINED:
            return jsonify({"error": "Model not trained"}), 403
        
        if recommender.model is None:
            recommender.load_model(Config.MODEL_PATH)
        
        recommendations = recommender.get_recommendations(user_id)
        return jsonify(recommendations), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()