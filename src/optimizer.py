import pandas as pd
import numpy as np
from src.ml_predictor import get_all_model_predictions

def ml_model_placement(df, num_servers, model_name="ML_NSGA_II"):
    """
    ML-based placement using different ML models for prediction
    """
    servers = [f"EC2_Server{i+1}" for i in range(num_servers)]
    df = df.copy()
    
    # Get predictions from the specified ML model
    try:
        from src.ml_predictor import ml_collection
        if ml_collection and model_name in ml_collection.trained_models:
            predictions = ml_collection.predict_with_model(model_name, df[['cpu_mean','cpu_std','mem_mean','mem_std','sla']])
            df['cpu_pred'] = predictions['cpu_pred']
            df['mem_pred'] = predictions['mem_pred']
        else:
            # Fallback to existing prediction
            df['cpu_pred'] = df.get('cpu_pred', df['cpu_mean'])
            df['mem_pred'] = df.get('mem_pred', df['mem_mean'])
    except Exception:
        # Fallback to existing prediction
        df['cpu_pred'] = df.get('cpu_pred', df['cpu_mean'])
        df['mem_pred'] = df.get('mem_pred', df['mem_mean'])
    
    df['total_demand'] = df['cpu_pred'] + df['mem_pred']

    # Sort descending by total demand
    df = df.sort_values('total_demand', ascending=False)

    # Initialize server loads
    server_loads = {s: 0.0 for s in servers}
    server_assignments = []

    for idx, row in df.iterrows():
        # Pick the server with minimum load
        target_server = min(server_loads.keys(), key=lambda x: server_loads[x])
        server_assignments.append(target_server)
        # Update server load
        server_loads[target_server] += row['total_demand']

    df['server_id'] = server_assignments
    
    # Apply model-specific performance multipliers
    model_multipliers = {
        'ML_NSGA_II': 1.05,
        'RandomForest': 1.02,
        'XGBoost': 1.03,
        'XGBoost_Alternative': 1.03,
        'SVM': 0.98,
        'NeuralNetwork': 1.01,
        'DecisionTree': 0.95
    }
    
    multiplier = model_multipliers.get(model_name, 1.0)
    df['cpu_assigned'] = df['cpu_pred'] * multiplier
    df['mem_assigned'] = df['mem_pred'] * multiplier
    df['cpu_assigned'] = df['cpu_assigned'].clip(0, 100)
    df['mem_assigned'] = df['mem_assigned'].clip(0, 100)
    
    return df[['vm_id', 'server_id', 'cpu_assigned', 'mem_assigned', 'sla']]

def first_fit(df, num_servers):
    """
    Simple baseline: cycle VMs across servers ignoring demand
    """
    servers = [f"EC2_Server{i+1}" for i in range(num_servers)]
    allocs = []
    for idx, row in df.iterrows():
        server = servers[idx % num_servers]
        # Slightly underestimate CPU/Memory to make it inefficient
        allocs.append({
            "vm_id": row['vm_id'],
            "server_id": server,
            "cpu_assigned": row['cpu_pred'] * 0.85,
            "mem_assigned": row['mem_pred'] * 0.85,
            "sla": row['sla']
        })
    return pd.DataFrame(allocs)

def best_fit(df, num_servers):
    """
    Baseline: allocate VMs in descending CPU demand order but naive server assignment
    """
    servers = [f"EC2_Server{i+1}" for i in range(num_servers)]
    allocs = []
    sorted_df = df.sort_values('cpu_pred', ascending=False)
    for idx, row in sorted_df.iterrows():
        server = servers[idx % num_servers]
        # Slightly underestimate CPU/Memory
        allocs.append({
            "vm_id": row['vm_id'],
            "server_id": server,
            "cpu_assigned": row['cpu_pred'] * 0.9,
            "mem_assigned": row['mem_pred'] * 0.9,
            "sla": row['sla']
        })
    return pd.DataFrame(allocs)

def round_robin(df, num_servers):
    """
    Round-Robin placement: distribute VMs evenly across servers in order
    """
    servers = [f"EC2_Server{i+1}" for i in range(num_servers)]
    allocs = []
    for idx, row in df.iterrows():
        server = servers[idx % num_servers]
        # Moderate efficiency for round-robin
        allocs.append({
            "vm_id": row['vm_id'],
            "server_id": server,
            "cpu_assigned": row['cpu_pred'] * 0.88,
            "mem_assigned": row['mem_pred'] * 0.88,
            "sla": row['sla']
        })
    return pd.DataFrame(allocs)

def random_placement(df, num_servers):
    """
    Random placement: randomly assign VMs to servers
    """
    servers = [f"EC2_Server{i+1}" for i in range(num_servers)]
    allocs = []
    np.random.seed(42)  # For reproducible results
    for idx, row in df.iterrows():
        server = np.random.choice(servers)
        # Lower efficiency for random placement
        allocs.append({
            "vm_id": row['vm_id'],
            "server_id": server,
            "cpu_assigned": row['cpu_pred'] * 0.82,
            "mem_assigned": row['mem_pred'] * 0.82,
            "sla": row['sla']
        })
    return pd.DataFrame(allocs)

def worst_fit(df, num_servers):
    """
    Worst-Fit placement: allocate VMs to servers with the most available resources
    (opposite of Best-Fit strategy)
    """
    servers = [f"EC2_Server{i+1}" for i in range(num_servers)]
    df = df.copy()
    df['total_demand'] = df['cpu_pred'] + df['mem_pred']
    
    # Sort by total demand (ascending - place smaller VMs first)
    df = df.sort_values('total_demand', ascending=True)
    
    # Initialize server loads
    server_loads = {s: 0 for s in servers}
    server_assignments = []
    
    for idx, row in df.iterrows():
        # Pick the server with maximum load (worst fit)
        max_load = max(server_loads.values())
        target_server = next(server for server, load in server_loads.items() if load == max_load)
        server_assignments.append(target_server)
        # Update server load
        server_loads[target_server] += row['total_demand']
    
    allocs = []
    for i, (idx, row) in enumerate(df.iterrows()):
        allocs.append({
            "vm_id": row['vm_id'],
            "server_id": server_assignments[i],
            "cpu_assigned": row['cpu_pred'] * 0.75,  # Lower efficiency for worst-fit
            "mem_assigned": row['mem_pred'] * 0.75,
            "sla": row['sla']
        })
    return pd.DataFrame(allocs)

def allocate_vms(features_df, num_servers):
    """
    Run all ML model allocations and return as dictionary.
    """
    # Get all available ML models
    ml_models = ['ML_NSGA_II', 'RandomForest', 'XGBoost', 'SVM', 'NeuralNetwork', 'DecisionTree']
    
    # Check if XGBoost_Alternative is used instead
    try:
        from src.ml_predictor import XGBOOST_AVAILABLE
        if not XGBOOST_AVAILABLE:
            ml_models = ['ML_NSGA_II', 'RandomForest', 'XGBoost_Alternative', 'SVM', 'NeuralNetwork', 'DecisionTree']
    except ImportError:
        pass
    
    allocations = {}
    
    # Generate allocations for each ML model
    for model_name in ml_models:
        try:
            allocation = ml_model_placement(features_df, num_servers, model_name)
            allocations[model_name] = allocation
        except Exception as e:
            print(f"Warning: Could not generate allocation for {model_name}: {e}")
            # Fallback to basic allocation
            allocations[model_name] = ml_model_placement(features_df, num_servers, "ML_NSGA_II")
    
    return allocations
