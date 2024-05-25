# -*- coding: utf-8 -*-
"""
Created on Sat May 25 07:32:11 2024

@author: Gregory_Guo
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd


########################################################
####################   Data Load   #####################
########################################################

# Load all necessary data for the first question analysis
load_data_path = r'C:/Users/29639/Desktop/2024电工杯/题目/1716500748329333/A题/附件1：各园区典型日负荷数据.xlsx'
generation_data_path = r'C:/Users/29639/Desktop/2024电工杯/题目/1716500748329333/A题/附件2：各园区典型日风光发电数据.xlsx'

# Load the load data
load_data = pd.read_excel(load_data_path)

# Load the wind and solar generation data
generation_data = pd.read_excel(generation_data_path, header=2)
generation_data.columns = ['时间（h）', '园区A光伏出力（p.u.）', '园区B风电出力（p.u.）', '园区C光伏出力（p.u.）', '园区C风电出力（p.u.）']

# Define the installed capacities
installed_capacity = {
    '园区A光伏': 750,  # kW
    '园区B风电': 1000,  # kW
    '园区C光伏': 600,  # kW
    '园区C风电': 500   # kW
}

# Calculate actual generation values
generation_data['园区A光伏出力(kW)'] = generation_data['园区A光伏出力（p.u.）'] * installed_capacity['园区A光伏']
generation_data['园区B风电出力(kW)'] = generation_data['园区B风电出力（p.u.）'] * installed_capacity['园区B风电']
generation_data['园区C光伏出力(kW)'] = generation_data['园区C光伏出力（p.u.）'] * installed_capacity['园区C光伏']
generation_data['园区C风电出力(kW)'] = generation_data['园区C风电出力（p.u.）'] * installed_capacity['园区C风电']

# Drop the original per-unit columns
generation_data_actual = generation_data.drop(columns=['园区A光伏出力（p.u.）', '园区B风电出力（p.u.）', '园区C光伏出力（p.u.）', '园区C风电出力（p.u.）'])
# Combine load and generation data for each park
combined_data = pd.merge(load_data, generation_data_actual, on='时间（h）')
# 将时间列转换为datetime格式
combined_data['时间（h）'] = pd.to_datetime(combined_data['时间（h）'], format='%H:%M:%S')

# 汇总联合园区的负荷和发电数据
combined_data['联合园区负荷(kW)'] = combined_data[['园区A负荷(kW)', '园区B负荷(kW)', '园区C负荷(kW)']].sum(axis=1)
combined_data['联合园区光伏出力(kW)'] = combined_data[['园区A光伏出力(kW)', '园区C光伏出力(kW)']].sum(axis=1)
combined_data['联合园区风电出力(kW)'] = combined_data[['园区B风电出力(kW)', '园区C风电出力(kW)']].sum(axis=1)

# 创建模型
model = gp.Model("energy_optimization")

# 时间步长
time_steps = combined_data.shape[0]

# 创建变量
storage_charge = model.addVars(time_steps, vtype=GRB.BINARY, name="storage_charge")
storage_discharge = model.addVars(time_steps, vtype=GRB.BINARY, name="storage_discharge")
storage_energy = model.addVars(time_steps, vtype=GRB.CONTINUOUS, name="storage_energy")
storage_power_charge = model.addVars(time_steps, vtype=GRB.CONTINUOUS, name="storage_power_charge")
storage_power_discharge = model.addVars(time_steps, vtype=GRB.CONTINUOUS, name="storage_power_discharge")
purchase = model.addVars(time_steps, vtype=GRB.CONTINUOUS, name="purchase")
curtailment = model.addVars(time_steps, vtype=GRB.CONTINUOUS, name="curtailment")

# 新增的储能容量和储能功率决策变量
storage_capacity = model.addVar(vtype=GRB.CONTINUOUS, name="storage_capacity")
storage_power = model.addVar(vtype=GRB.CONTINUOUS, name="storage_power")

# 初始储能状态
initial_storage_energy = 50  # 初始储能能量
efficiency_charge = 0.95  # 充电效率
efficiency_discharge = 0.95  # 放电效率
purchase_cost = 1  # 购电成本
pv_cost = 0.4  # 光伏发电成本
wind_cost = 0.5  # 风电发电成本

# 储能系统的成本参数
power_cost_per_kw = 800  # 元/kW
energy_cost_per_kwh = 1800  # 元/kWh

# 约束条件
for t in range(time_steps):
    load = combined_data.at[t, '联合园区负荷(kW)']
    pv_gen = combined_data.at[t, '联合园区光伏出力(kW)']
    wind_gen = combined_data.at[t, '联合园区风电出力(kW)']
    
    # 功率平衡约束
    model.addConstr(load == pv_gen + wind_gen - curtailment[t] + purchase[t] - storage_power_discharge[t] + storage_power_charge[t])

    # 储能不能同时充电和放电
    model.addConstr(storage_charge[t] + storage_discharge[t] <= 1)
    
    # 储能充放电功率限制
    model.addConstr(storage_power_charge[t] <= storage_charge[t] * storage_power)
    model.addConstr(storage_power_discharge[t] <= storage_discharge[t] * storage_power)
    
    # 储能状态约束
    if t == 0:
        model.addConstr(storage_energy[t] == initial_storage_energy + storage_power_charge[t] * efficiency_charge - storage_power_discharge[t] / efficiency_discharge)
    else:
        model.addConstr(storage_energy[t] == storage_energy[t-1] + storage_power_charge[t] * efficiency_charge - storage_power_discharge[t] / efficiency_discharge)
    
    # 储能能量约束
    model.addConstr(storage_energy[t] <= storage_capacity)
    model.addConstr(storage_energy[t] >= 0)

    # 购电量和弃电量非负约束
    model.addConstr(purchase[t] >= 0)
    model.addConstr(curtailment[t] >= 0)

# 初始和终端储能状态约束
model.addConstr(storage_energy[0] == initial_storage_energy)
model.addConstr(storage_energy[time_steps - 1] == initial_storage_energy)

# 目标函数：最小化总成本（包括购电成本、发电成本和储能系统的投资成本）
total_cost = gp.quicksum(purchase[t] * purchase_cost + pv_gen * pv_cost + wind_gen * wind_cost for t in range(time_steps))
total_investment_cost = storage_power * power_cost_per_kw + storage_capacity * energy_cost_per_kwh
model.setObjective(total_cost + total_investment_cost, GRB.MINIMIZE)

# 求解模型
model.optimize()

# 打印结果
if model.status == GRB.OPTIMAL:
    print(f'Optimal objective: {model.objVal}')
    for t in range(time_steps):
        print(f'Time {t}: Purchase: {purchase[t].x}, Curtailment: {curtailment[t].x}, Storage Charge: {storage_charge[t].x}, Storage Discharge: {storage_discharge[t].x}, Storage Energy: {storage_energy[t].x}')
    print(f'Optimal Storage Capacity: {storage_capacity.x} kWh')
    print(f'Optimal Storage Power: {storage_power.x} kW')
else:
    print('No optimal solution found.')
