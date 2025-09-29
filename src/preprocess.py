import pandas as pd

def preprocess_workload(df):
    """
    Aggregate per-VM stats for placement.
    """
    agg_list = []
    vm_ids = df['vm_id'].unique()
    for vm in vm_ids:
        vm_data = df[df['vm_id'] == vm].sort_values('timestep')
        agg_list.append({
            "vm_id": vm,
            "cpu_mean": vm_data['cpu'].mean(),
            "cpu_std": vm_data['cpu'].std(),
            "mem_mean": vm_data['mem'].mean(),
            "mem_std": vm_data['mem'].std(),
            "cpu_next": vm_data['cpu'].iloc[-1],
            "mem_next": vm_data['mem'].iloc[-1],
            "sla": vm_data['sla'].iloc[0]
        })
    return pd.DataFrame(agg_list)