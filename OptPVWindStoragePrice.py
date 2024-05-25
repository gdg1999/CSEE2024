# -*- coding: utf-8 -*-
"""
Created on Sat May 25 07:41:44 2024

@author: Gregory_Guo
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd


########################################################
####################   Data Load   #####################
########################################################
# Define the installed capacities
installed_capacity = {
    '园区A光伏': 750,  # kW
    '园区B风电': 1000,  # kW
    '园区C光伏': 600,  # kW
    '园区C风电': 500   # kW
}

# Load all necessary data for the first question analysis
load_data_path = r'C:/Users/29639/Desktop/2024电工杯/题目/1716500748329333/A题/附件1：各园区典型日负荷数据.xlsx'
generation_monthly_data_path = r'C:/Users/29639/Desktop/2024电工杯/题目/1716500748329333/A题/附件3：12个月各园区典型日风光发电数据.xlsx'


# Load the load data
load_data = pd.read_excel(load_data_path)

# Load the wind and solar generation data
generation_monthly_data = pd.read_excel(generation_monthly_data_path, header=2, sheet_name=0)


# 遍历所有工作表
for sheet_name in pd.ExcelFile(generation_monthly_data_path).sheet_names:
    # 读取当前工作表的数据
    generation_monthly_data = pd.read_excel(generation_monthly_data_path, header=2, sheet_name=sheet_name)

    generation_monthly_data.columns = ['时间（h）', f'{sheet_name}园区A光伏出力（p.u.）', f'{sheet_name}园区B风电出力（p.u.）', f'{sheet_name}园区C光伏出力（p.u.）', f'{sheet_name}园区C风电出力（p.u.）']

    # Calculate actual generation values
    generation_monthly_data[f'{sheet_name}园区A光伏出力(kW)'] = generation_monthly_data[f'{sheet_name}园区A光伏出力（p.u.）'] * installed_capacity['园区A光伏']
    generation_monthly_data[f'{sheet_name}园区B风电出力(kW)'] = generation_monthly_data[f'{sheet_name}园区B风电出力（p.u.）'] * installed_capacity['园区B风电']
    generation_monthly_data[f'{sheet_name}园区C光伏出力(kW)'] = generation_monthly_data[f'{sheet_name}园区C光伏出力（p.u.）'] * installed_capacity['园区C光伏']
    generation_monthly_data[f'{sheet_name}园区C风电出力(kW)'] = generation_monthly_data[f'{sheet_name}园区C风电出力（p.u.）'] * installed_capacity['园区C风电']

    # Drop the original per-unit columns
    generation_monthly_data_actual = generation_monthly_data.drop(columns=[f'{sheet_name}园区A光伏出力（p.u.）', f'{sheet_name}园区B风电出力（p.u.）', f'{sheet_name}园区C光伏出力（p.u.）', f'{sheet_name}园区C风电出力（p.u.）'])

    # Combine load and generation data for each park
    load_data = pd.merge(load_data, generation_monthly_data_actual, on='时间（h）')

combined_data = load_data
# 将时间列转换为datetime格式（假设时间列的名称为'时间'）
combined_data['时间（h）'] = pd.to_datetime(combined_data['时间（h）'], format='%H:%M:%S').dt.hour

# 分时电价设置
def get_time_of_use_price(time):
    # 假设高峰电价为1.0元/kWh，低谷电价为0.4元/kWh
    peak_price = 1.0
    off_peak_price = 0.4

    # 定义高峰、平时和低谷时间段
    if time in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
        return peak_price
    else:
        return off_peak_price

# 应用分时电价到每个时间点
combined_data['电价'] = combined_data['时间（h）'].apply(get_time_of_use_price)

# 创建模型
model = gp.Model("wind_solar_storage_optimization")

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

# 新增的储能容量、储能功率、风电和光伏的容量决策变量
storage_capacity = model.addVar(vtype=GRB.CONTINUOUS, name="storage_capacity")
storage_power = model.addVar(vtype=GRB.CONTINUOUS, name="storage_power")
wind_capacity = model.addVar(vtype=GRB.CONTINUOUS, name="wind_capacity")
solar_capacity = model.addVar(vtype=GRB.CONTINUOUS, name="solar_capacity")

# 初始储能状态
initial_storage_energy = 50  # 初始储能能量
efficiency_charge = 0.95  # 充电效率
efficiency_discharge = 0.95  # 放电效率
pv_cost = 0.4  # 光伏发电成本
wind_cost = 0.5  # 风电发电成本

# 系统的成本参数
power_cost_per_kw = 800  # 储能功率成本，元/kW
energy_cost_per_kwh = 1800  # 储能容量成本，元/kWh
wind_cost_per_kw = 3000  # 风电容量成本，元/kW
solar_cost_per_kw = 2500  # 光伏容量成本，元/kW

# 约束条件
for t in range(time_steps):
    load = combined_data.at[t, '联合园区负荷(kW)']
    pv_gen = combined_data.at[t, '联合园区光伏出力(kW)'] * solar_capacity
    wind_gen = combined_data.at[t, '联合园区风电出力(kW)'] * wind_capacity
    
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

# 目标函数：最小化总成本（包括购电成本、发电成本和系统的投资成本）
total_cost = gp.quicksum(purchase[t] * combined_data.at[t, '电价'] + 
                         combined_data.at[t, '联合园区光伏出力(kW)'] * solar_capacity * pv_cost + 
                         combined_data.at[t, '联合园区风电出力(kW)'] * wind_capacity * wind_cost 
                         for t in range(time_steps))

total_investment_cost = storage_power * power_cost_per_kw + \
                        storage_capacity * energy_cost_per_kwh + \
                        wind_capacity * wind_cost_per_kw + \
                        solar_capacity * solar_cost_per_kw

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
    print(f'Optimal Wind Capacity: {wind_capacity.x} kW')
    print(f'Optimal Solar Capacity: {solar_capacity.x} kW')
else:
    print('No optimal solution found.')
