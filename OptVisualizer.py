# -*- coding: utf-8 -*-
"""
Created on Sat May 25 13:25:16 2024

@author: Gregory_Guo
"""

def storage_operation(row, previous_soc, load_col, gen_col):
    """
    储能运行策略
    """
    
    load = row[load_col]
    if isinstance(gen_col, list):
        generation = sum(row[col] for col in gen_col)
    else:
        generation = row[gen_col]
    net_load = load - generation
    if net_load > 0:
        # 需要从储能放电
        charge = 0
        discharge = min(storage_power, net_load / efficiency, previous_soc - soc_min)
        new_soc = previous_soc - discharge
        net_load -= discharge * efficiency
    else:
        # 需要给储能充电
        discharge = 0
        charge = min(storage_power, -net_load * efficiency, soc_max - previous_soc)
        new_soc = previous_soc + charge
        net_load += charge / efficiency
        
    return new_soc, net_load, discharge, charge