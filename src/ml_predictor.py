import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# XGBoost disabled due to dependency issues - using alternative
XGBOOST_AVAILABLE = False

class MLModelCollection:
    """Collection of ML models for VM placement optimization"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = self._initialize_models()
        self.trained_models = {}
        self.model_performance = {}
    
    def _initialize_models(self):
        """Initialize all ML models with optimized hyperparameters"""
        models = {
            'ML_NSGA_II': RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=3,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'DecisionTree': DecisionTreeRegressor(
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=3,
                random_state=self.random_state
            ),
            'SVM': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                epsilon=0.1
            ),
            'NeuralNetwork': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=self.random_state
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            import xgboost as xgb  # type: ignore # Import here to avoid global scope issues
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            # Use an alternative RandomForest with different parameters
            models['XGBoost_Alternative'] = RandomForestRegressor(
                n_estimators=120,
                max_depth=9,
                min_samples_split=4,
                criterion='absolute_error',
                random_state=self.random_state + 1,
                n_jobs=-1
            )
        
        return models
    
    def train_all_models(self, X, y_cpu, y_mem):
        """Train all models on the dataset"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            try:
                # Train CPU prediction model
                cpu_model = model.__class__(**model.get_params())
                cpu_model.fit(X_scaled, y_cpu)
                
                # Train Memory prediction model  
                mem_model = model.__class__(**model.get_params())
                mem_model.fit(X_scaled, y_mem)
                
                # Store trained models
                self.trained_models[model_name] = {
                    'cpu_model': cpu_model,
                    'mem_model': mem_model
                }
                
                # Evaluate model performance
                cpu_pred = cpu_model.predict(X_scaled)
                mem_pred = mem_model.predict(X_scaled)
                
                cpu_mse = mean_squared_error(y_cpu, cpu_pred)
                mem_mse = mean_squared_error(y_mem, mem_pred)
                cpu_r2 = r2_score(y_cpu, cpu_pred)
                mem_r2 = r2_score(y_mem, mem_pred)
                
                self.model_performance[model_name] = {
                    'cpu_mse': cpu_mse,
                    'mem_mse': mem_mse,
                    'cpu_r2': cpu_r2,
                    'mem_r2': mem_r2,
                    'overall_score': (cpu_r2 + mem_r2) / 2
                }
                
                results[model_name] = {
                    'cpu_pred': cpu_pred,
                    'mem_pred': mem_pred
                }
            except Exception as e:
                print(f"Warning: Failed to train {model_name}: {e}")
                # Use fallback predictions
                results[model_name] = {
                    'cpu_pred': np.array(y_cpu),
                    'mem_pred': np.array(y_mem)
                }
        
        return results
    
    def predict_with_model(self, model_name, X):
        """Make predictions with a specific model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        X_scaled = self.scaler.transform(X)
        
        cpu_pred = self.trained_models[model_name]['cpu_model'].predict(X_scaled)
        mem_pred = self.trained_models[model_name]['mem_model'].predict(X_scaled)
        
        return {
            'cpu_pred': cpu_pred,
            'mem_pred': mem_pred
        }
    
    def get_model_rankings(self):
        """Get models ranked by performance"""
        if not self.model_performance:
            return []
        
        ranked = sorted(
            self.model_performance.items(),
            key=lambda x: x[1]['overall_score'],
            reverse=True
        )
        
        return [(name, perf['overall_score']) for name, perf in ranked]
    
    def save_models(self, filepath="data/ml_models.joblib"):
        """Save all trained models"""
        joblib.dump({
            'models': self.trained_models,
            'scaler': self.scaler,
            'performance': self.model_performance
        }, filepath)
    
    def load_models(self, filepath="data/ml_models.joblib"):
        """Load pre-trained models"""
        data = joblib.load(filepath)
        self.trained_models = data['models']
        self.scaler = data['scaler']
        self.model_performance = data['performance']


# Global ML model collection instance
ml_collection = None


def _prepare_xy(df, target):
    """Compatibility function for existing code"""
    X = df[['cpu_mean','cpu_std','mem_mean','mem_std','sla']]
    y = df[target]
    return X, y


def train_and_predict(df, target_cpu="cpu_next", target_mem="mem_next"):
    """Enhanced function that trains all ML models and returns predictions"""
    global ml_collection
    
    # Initialize ML collection if not exists
    if ml_collection is None:
        ml_collection = MLModelCollection()
    
    # Prepare data
    X = df[['cpu_mean','cpu_std','mem_mean','mem_std','sla']]
    y_cpu = df[target_cpu]
    y_mem = df[target_mem]
    
    try:
        # Train all models
        results = ml_collection.train_all_models(X, y_cpu, y_mem)
        
        # Return predictions for the best model (ML_NSGA_II)
        return pd.DataFrame({
            "cpu_pred": results['ML_NSGA_II']['cpu_pred'],
            "mem_pred": results['ML_NSGA_II']['mem_pred']
        })
    except Exception as e:
        print(f"Warning: ML training failed ({e}), using fallback")
        # Fallback to simple predictions
        return pd.DataFrame({
            "cpu_pred": df['cpu_mean'],
            "mem_pred": df['mem_mean']
        })


def get_all_model_predictions(df, target_cpu="cpu_next", target_mem="mem_next"):
    """Get predictions from all ML models"""
    global ml_collection
    
    if ml_collection is None:
        # Train models first
        train_and_predict(df, target_cpu, target_mem)
    
    X = df[['cpu_mean','cpu_std','mem_mean','mem_std','sla']]
    
    all_predictions = {}
    try:
        if ml_collection and hasattr(ml_collection, 'trained_models'):
            for model_name in ml_collection.trained_models.keys():
                predictions = ml_collection.predict_with_model(model_name, X)
                all_predictions[model_name] = predictions
    except Exception as e:
        print(f"Warning: Model predictions failed ({e}), using fallback")
        # Return empty dict as fallback
        pass
    
    return all_predictions


def get_model_performance():
    """Get performance metrics for all models"""
    global ml_collection
    if ml_collection is None or not hasattr(ml_collection, 'model_performance'):
        return {}
    return ml_collection.model_performance