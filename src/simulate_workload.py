import pandas as pd
import numpy as np

def generate_workload(num_vms=10, timesteps=10):
    """
    Generates synthetic AWS EC2-like workload with realistic VM patterns.
    Each VM has a type: CPU, Memory, or Balanced, and SLA correlated with type.
    """
    rows = []

    # Assign VM types and SLA tiers
    vm_types = {}
    sla_values = {}
    for vm in range(1, num_vms+1):
        vm_type = np.random.choice(['CPU', 'MEM', 'BALANCED'])
        vm_types[f"VM{vm}"] = vm_type
        sla_values[f"VM{vm}"] = np.random.uniform(0.95, 1.0)  # high SLA for all

    for t in range(1, timesteps+1):
        for vm in range(1, num_vms+1):
            vm_id = f"VM{vm}"
            vm_type = vm_types[vm_id]

            # Generate realistic CPU & memory usage per type
            if vm_type == 'CPU':
                cpu = np.random.uniform(50, 90) + np.sin(t / 2)*10
                mem = np.random.uniform(10, 50)
            elif vm_type == 'MEM':
                cpu = np.random.uniform(10, 50)
                mem = np.random.uniform(50, 90) + np.sin(t / 3)*10
            else:  # BALANCED
                cpu = np.random.uniform(30, 70)
                mem = np.random.uniform(30, 70)

            rows.append({
                "vm_id": vm_id,
                "timestep": t,
                "cpu": min(cpu, 100),
                "mem": min(mem, 100),
                "sla": sla_values[vm_id],
                "vm_type": vm_type
            })

    df = pd.DataFrame(rows)
    return df
