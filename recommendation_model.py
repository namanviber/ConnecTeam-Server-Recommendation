import numpy as np
import pandas as pd
import pickle
import warnings
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix

warnings.filterwarnings('ignore')

class RecommendationSystem:
    def __init__(self):
        self.dataset = None
        self.model = None
        self.user_features_matrix = None
        self.server_features_matrix = None
        self.interactions = None
        self.df_user = None
        self.df_server = None
        self.df_interaction = None
        self.category_mapping = None
        self.user_map = None
        self.server_map = None

    def prepare_data(self, df_interaction, df_server, df_user, df_category):
        """Preprocess and prepare data for training"""
        try:
            self.df_interaction = df_interaction
            self.df_server = df_server
            self.df_user = df_user            

            self.category_mapping = set(df_category['Category ID'])
            self.user_map = self.df_user.set_index('user_id')['user_num']
            self.server_map = self.df_server.set_index('server_id')['server_num']
            
            self.df_interaction['user_num'] = self.df_interaction['user_id'].map(self.user_map)
            self.df_interaction['server_num'] = self.df_interaction['server_id'].map(self.server_map)
            self.df_interaction['rating'] = self.df_interaction['rating'].fillna(0).astype(float)
            
            return True
        except Exception as e:
            print(f"Error in data preparation: {e}")
            return False


    def train_model(self):
        """Train the recommendation model"""
        try:
            # Prepare feature matrices
            server_features = [
                (row['server_num'], row['server_category']) 
                for _, row in self.df_server.iterrows()
            ]
            user_features = [
                (row['user_num'], row['server_category']) 
                for _, row in self.df_user.iterrows()
            ]

            # Create dataset
            self.dataset = Dataset()
            self.dataset.fit(
                users=self.df_user['user_num'].unique(),
                items=self.df_server['server_num'].unique(),
                user_features=self.category_mapping,
                item_features=self.category_mapping,
            )

            # Build interactions
            self.interactions, weights = self.dataset.build_interactions([
                (row['user_num'], row['server_num'], row['rating']) 
                for _, row in self.df_interaction.iterrows()
            ])

            # Build feature matrices
            self.user_features_matrix = self.dataset.build_user_features(user_features)
            self.server_features_matrix = self.dataset.build_item_features(server_features)

            # Train model
            self.model = LightFM(
                no_components=150, 
                learning_rate=0.05, 
                loss='warp', 
                random_state=42
            )

            self.model.fit(
                self.interactions, 
                item_features=self.server_features_matrix,
                user_features=self.user_features_matrix, 
                sample_weight=weights, 
                epochs=30, 
                num_threads=1
            )

            precision = precision_at_k(
                self.model, self.interactions, k=5, 
                user_features=self.user_features_matrix, 
                item_features=self.server_features_matrix
            ).mean()
            recall = recall_at_k(
                self.model, self.interactions, k=5, 
                user_features=self.user_features_matrix, 
                item_features=self.server_features_matrix
            ).mean()
            auc = auc_score(
                self.model, self.interactions, 
                user_features=self.user_features_matrix, 
                item_features=self.server_features_matrix
            ).mean()

            print(f'Precision@k: {precision:.4f}')
            print(f'Recall@k: {recall:.4f}')
            print(f'AUC: {auc:.4f}')

            return True
        except KeyError as e:
            print(f"Missing column in data: {e}")
            return False
        except Exception as e:
            print(f"Error in model training: {e}")
            return False


    def get_recommendations(self, user_id, top_k=10):
        """Get top K recommendations for a user"""
        try:
            user_num = self.user_map[user_id]

            all_server_scores = self.model.predict(
                user_ids=user_num, 
                item_ids=np.arange(len(self.df_server)),
                user_features=self.user_features_matrix, 
                item_features=self.server_features_matrix
            )

            top_server_indices = np.argsort(-all_server_scores)[:top_k]
            recommended_servers = self.df_server.loc[
                self.df_server['server_num'].isin(top_server_indices), 
                ['server_id', 'server_name', 'server_category']
            ]

            return recommended_servers.to_dict(orient='records')
        except Exception as e:
            print(f"Error in recommendations: {e}")
            return []

    def save_model(self, path):
        """Save trained model to file"""
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'dataset': self.dataset,
                    'user_features_matrix': self.user_features_matrix,
                    'server_features_matrix': self.server_features_matrix,
                    'interactions': self.interactions,
                    'user_map': self.user_map,
                    'server_map': self.server_map
                }, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, path):
        """Load trained model from file"""
        try:
            with open(path, 'rb') as f:
                saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.dataset = saved_data['dataset']
                self.user_features_matrix = saved_data['user_features_matrix']
                self.server_features_matrix = saved_data['server_features_matrix']
                self.interactions = saved_data['interactions']
                self.user_map = saved_data['user_map']
                self.server_map = saved_data['server_map']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False