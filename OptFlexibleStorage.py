# -*- coding: utf-8 -*-
"""
Created on Sat May 25 07:32:11 2024

@author: Gregory_Guo
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use( ['science',"grid","ieee"])

########################################################
####################   Data Load   #####################
########################################################

# Load all necessary data for the first question analysis
load_data_path = r'C:/Users/29639/Desktop/2024电工杯/题目/1716500748329333/A题/附件1：各园区典型日负荷数据.xlsx'
generation_data_path = r'C:/Users/29639/Desktop/2024电工杯/题目/1716500748329333/A题/附件2：各园区典型日风光发电数据.xlsx'

# Load the load data
load_data = pd.read_excel(load_data_path)
load_data.columns = ['时间（h）', 'A园区负荷(kW)', 'B园区负荷(kW)', 'C园区负荷(kW)']


# Load the wind and solar generation data
generation_data = pd.read_excel(generation_data_path, header=2)
generation_data.columns = ['时间（h）', 'A园区光伏出力（p.u.）', 'B园区风电出力（p.u.）', 'C园区光伏出力（p.u.）', 'C园区风电出力（p.u.）']

# Define the installed capacities
installed_capacity = {
    '园区A光伏': 750,  # kW
    '园区B风电': 1000,  # kW
    '园区C光伏': 600,  # kW
    '园区C风电': 500   # kW
}

# Calculate actual generation values
generation_data['A园区光伏出力(kW)'] = generation_data['A园区光伏出力（p.u.）'] * installed_capacity['园区A光伏']
generation_data['B园区风电出力(kW)'] = generation_data['B园区风电出力（p.u.）'] * installed_capacity['园区B风电']
generation_data['C园区光伏出力(kW)'] = generation_data['C园区光伏出力（p.u.）'] * installed_capacity['园区C光伏']
generation_data['C园区风电出力(kW)'] = generation_data['C园区风电出力（p.u.）'] * installed_capacity['园区C风电']

# Drop the original per-unit columns
generation_data_actual = generation_data.drop(columns=['A园区光伏出力（p.u.）', 'B园区风电出力（p.u.）', 'C园区光伏出力（p.u.）', 'C园区风电出力（p.u.）'])
# Combine load and generation data for each park
combined_data = pd.merge(load_data, generation_data_actual, on='时间（h）')
# 将时间列转换为datetime格式
combined_data['时间（h）'] = pd.to_datetime(combined_data['时间（h）'], format='%H:%M:%S').dt.hour

# 汇总联合园区的负荷和发电数据
combined_data['联合园区负荷(kW)'] = combined_data[['A园区负荷(kW)', 'B园区负荷(kW)', 'C园区负荷(kW)']].sum(axis=1)
combined_data['联合园区光伏出力(kW)'] = combined_data[['A园区光伏出力(kW)', 'C园区光伏出力(kW)']].sum(axis=1)
combined_data['联合园区风电出力(kW)'] = combined_data[['B园区风电出力(kW)', 'C园区风电出力(kW)']].sum(axis=1)


combined_data['A园区风电出力(kW)'] = 0
combined_data['B园区光伏出力(kW)'] = 0

# 初始储能状态
initial_storage_energy = 0.3  # 初始储能能量%
efficiency_charge = 0.95  # 充电效率
efficiency_discharge = 0.95  # 放电效率
purchase_cost = 1  # 购电成本
pv_cost = 0.4  # 光伏发电成本
wind_cost = 0.5  # 风电发电成本

# 储能系统的成本参数
power_cost_per_kw = 800  # 元/kW
energy_cost_per_kwh = 1800  # 元/kWh

########################################################
####################   Opt Model   #####################
########################################################

def OptModel(modelCase, combined_data, park):

    # 创建模型
    model = gp.Model(modelCase)
    
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
    
    # 约束条件
    for t in range(time_steps):
        load = combined_data.at[t, f'{park}园区负荷(kW)']
        pv_gen = combined_data.at[t, f'{park}园区光伏出力(kW)']
        wind_gen = combined_data.at[t, f'{park}园区风电出力(kW)']
        
        # 功率平衡约束
        model.addConstr(load == pv_gen + wind_gen - curtailment[t] + purchase[t] + storage_power_discharge[t] - storage_power_charge[t])
    
        # 储能不能同时充电和放电
        model.addConstr(storage_charge[t] + storage_discharge[t] <= 1)
        
        # 储能充放电功率限制
        model.addConstr(storage_power_charge[t] <= storage_charge[t] * storage_power)
        model.addConstr(storage_power_discharge[t] <= storage_discharge[t] * storage_power)
        
        # 储能状态约束
        if t == 0:
            model.addConstr(storage_energy[t] == initial_storage_energy*storage_capacity + storage_power_charge[t] * efficiency_charge - storage_power_discharge[t] / efficiency_discharge)
        else:
            model.addConstr(storage_energy[t] == storage_energy[t-1] + storage_power_charge[t] * efficiency_charge - storage_power_discharge[t] / efficiency_discharge)
        
        # 储能能量约束
        model.addConstr(storage_energy[t] <= 0.9 * storage_capacity)
        model.addConstr(storage_energy[t] >= 0.1 * storage_capacity)
    
        # 购电量和弃电量非负约束
        model.addConstr(purchase[t] >= 0)
        model.addConstr(curtailment[t] >= 0)
    
    # 初始和终端储能状态约束
    model.addConstr(storage_energy[0] == initial_storage_energy*storage_capacity)
    model.addConstr(storage_energy[time_steps - 1] == initial_storage_energy*storage_capacity)
    
    # 目标函数：最小化总成本（包括购电成本、发电成本和储能系统的投资成本）
    total_cost = gp.quicksum(purchase[t] * purchase_cost + pv_gen * pv_cost + wind_gen * wind_cost for t in range(time_steps))
    total_investment_cost = (storage_power * power_cost_per_kw + storage_capacity * energy_cost_per_kwh)/365*0.1295
    model.setObjective(total_cost + total_investment_cost, GRB.MINIMIZE)
    
    # 求解模型
    model.optimize()
    
    return (model,
            storage_charge, storage_discharge,
            storage_power_charge, storage_power_discharge,
            storage_energy, purchase, curtailment,
            storage_capacity, storage_power)
    
    # 打印结果
    # if model.status == GRB.OPTIMAL:
    #     print(f'Optimal objective: {model.objVal}')
    #     for t in range(time_steps):
    #         print(f'Time {t}: Purchase: {purchase[t].x}, Curtailment: {curtailment[t].x}, Storage Charge: {storage_charge[t].x}, Storage Discharge: {storage_discharge[t].x}, Storage Energy: {storage_energy[t].x}')
    #         print()
    #     print(f'Optimal Storage Capacity: {storage_capacity.x} kWh')
    #     print()
    #     print(f'Optimal Storage Power: {storage_power.x} kW')
    #     print()
    # else:
    #     print('No optimal solution found.')


def analyze_with_storage(park, load_col, gen_col, purchase_cost, generation_cost):
    """
    
    """
    
    # 初始化储能状态
    xInitial_storage = initial_storage_energy*storage_capacity.X


    # 运行储能策略
    for i in range(0, len(combined_data)):
        
        if i == 0:
            previous_soc = xInitial_storage

        else:
            previous_soc = combined_data.at[i-1, f'园区{park}_SOC (kWh)']

        combined_data.at[i, f'园区{park}_SOC (kWh)'] = storage_energy[i].X
        combined_data.at[i, f'{park}净负荷(kW)'] = purchase[i].X
        combined_data.at[i, f'{park}放电(kWh)'] = -storage_power_discharge[i].X
        combined_data.at[i, f'{park}充电(kWh)'] = storage_power_charge[i].X
        combined_data.at[i, f'{park}弃电量(kW)'] = curtailment[i].X
    # 经济性分析
    load = combined_data[load_col]
    
    if isinstance(gen_col, list):
        # 分开处理光伏和风电发电
        pv_generation = combined_data[gen_col[0]]
        wind_generation = combined_data[gen_col[1]]
        pv_cost = generation_cost[0]
        wind_cost = generation_cost[1]

        total_generation = pv_generation + wind_generation

        # 优先使用光伏发电
        used_pv_generation = pv_generation.copy()
        used_pv_generation[used_pv_generation > load] = load[used_pv_generation > load]
        remaining_load = load - used_pv_generation

        # 使用风电发电补充
        used_wind_generation = wind_generation.copy()
        used_wind_generation[used_wind_generation > remaining_load] = remaining_load[used_wind_generation > remaining_load]
        remaining_load -= used_wind_generation

        # 总的实际使用发电量
        total_used_generation = used_pv_generation + used_wind_generation

        # 弃风弃光量
        # excess_pv_generation = pv_generation - used_pv_generation
        # excess_wind_generation = wind_generation - used_wind_generation
        # excess_generation = excess_pv_generation + excess_wind_generation

        # 实际用电成本
        total_generation_cost = (used_pv_generation * pv_cost).sum() + (used_wind_generation * wind_cost).sum()

    else:
        total_generation = combined_data[gen_col]
        used_generation = total_generation.copy()
        used_generation[used_generation > load] = load[used_generation > load]
        remaining_load = load - used_generation

        # excess_generation = total_generation - used_generation
        # excess_generation[excess_generation < 0] = 0

        total_used_generation = used_generation
        total_generation_cost = (total_used_generation * generation_cost).sum()
        
        
    net_load = combined_data[f'{park}净负荷(kW)']
    purchaseList = net_load
    purchaseList[purchaseList < 0] = 0  # Set negative values to zero, as excess generation is not sold
    
    # excess_generation = total_generation - load - combined_data[f'{park}充电(kWh)']  # 弃电量要变化
    # excess_generation[excess_generation < 0] = 0  # Set negative values to zero
    
    excess_generation = combined_data[f'{park}弃电量(kW)']
    total_purchase_cost = (purchaseList * purchase_cost).sum()
    total_cost = total_purchase_cost + total_generation_cost
    avg_cost = total_cost / load.sum()
    
    return (purchaseList, excess_generation,
                                total_cost, avg_cost,
                                combined_data[f'园区{park}_SOC (kWh)'],
                                combined_data[f'{park}净负荷(kW)'],
                                combined_data[f'{park}放电(kWh)'],
                                combined_data[f'{park}充电(kWh)'])


def picture23(combined_data, park, storage_capacity):
    """
    
    """

    # 创建子图
    fig, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True)
    
    # 定义柱状图的宽度和透明度
    bar_width = 0.7
    alpha = 0.5
    
    # 字体设置
    title_fontsize = 26
    label_fontsize = 24
    legend_fontsize = 22
    tick_labelsize = 20
    
    # 绘制园区C的数据
    axes.bar(combined_data['时间（h）'], combined_data[f'{park}园区光伏出力(kW)'], width=bar_width, label='Joint Park PV Output (kW)', color='lightpink', alpha=alpha)
    axes.bar(combined_data['时间（h）'], combined_data[f'{park}园区风电出力(kW)'], width=bar_width, bottom= combined_data[f'{park}园区光伏出力(kW)'], label='Joint Park Wind Output (kW)', color='skyblue', alpha=alpha)
    bars_charge = axes.bar(combined_data['时间（h）'], combined_data[f'{park}园区充放电(kWh)'], width=bar_width, bottom= combined_data[f'{park}园区负荷(kW)'], label='Joint Park PV Charge (kW)', color='purple', alpha=1)
    axes.bar(combined_data['时间（h）'], combined_data[f'{park}园区负荷(kW)'], width=bar_width, label='Park C Load (kW)', edgecolor='black', facecolor='none', linewidth=2, linestyle = '--')
   
    axes.bar(combined_data['时间（h）'], combined_data[f'{park}园区购电量(kW)'], width=bar_width, bottom= combined_data[f'{park}园区光伏出力(kW)']+combined_data[f'{park}园区风电出力(kW)'], label='Park C Purchase (kW)', color='orange', alpha=alpha)
    axes.bar(combined_data['时间（h）'], combined_data[f'{park}园区弃电量(kW)'], width=bar_width, bottom = -combined_data[f'{park}园区弃电量(kW)'], label='Joint Park Excess (kW)', color='gray', alpha=alpha)
    
    axes.set_title('Park Load and PV/Wind Output', fontsize=title_fontsize)
    axes.set_xlabel('Time (h)', fontsize=label_fontsize)
    axes.set_ylabel('Power (kW)', fontsize=label_fontsize)
    # axes[2].legend(loc='upper right', fontsize=legend_fontsize)
    axes.tick_params(axis='both', which='major', labelsize=tick_labelsize)  # 调整刻度标签字体大小

    # 添加阴影效果
    for bar in bars_charge:
        bar.set_edgecolor('grey')
        bar.set_linewidth(1)
        bar.set_alpha(0.7)
    
    # 创建次坐标轴
    ax2 = axes.twinx()
    ax2.plot(combined_data['时间（h）'], combined_data[f'{park}园区_SOC(kWh)']/storage_capacity, label='Example Curve', color='green', marker='o')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('SOC (0-1)')
    ax2.tick_params(axis='y', which='major', labelsize=tick_labelsize)  # 调整刻度标签字体大小

    ax2.legend(loc='upper right')

    
    # 调整布局
    plt.tight_layout()
    plt.show()



(model,
 storage_charge, storage_discharge,
 storage_power_charge, storage_power_discharge,
 storage_energy, purchase, curtailment,
 storage_capacity, storage_power) = OptModel("energy_optimization", combined_data, '联合')
# 可视化
joint_result_without_storage = analyze_with_storage(
    "联合园区",
    '联合园区负荷(kW)', 
    ['联合园区光伏出力(kW)', '联合园区风电出力(kW)'], 
    1, 
    [0.4, 0.5]
)

combined_data['联合园区购电量(kW)'] = joint_result_without_storage[0]
combined_data['联合园区弃电量(kW)'] = joint_result_without_storage[1]
combined_data['联合园区_SOC(kWh)']= joint_result_without_storage[4]
combined_data['联合园区充放电(kWh)'] = combined_data[['联合园区放电(kWh)', '联合园区充电(kWh)']].sum(axis=1)


joint_purchase = joint_result_without_storage[0].sum()
joint_excess_generation = joint_result_without_storage[1].sum()
joint_total_cost = joint_result_without_storage[2]
joint_avg_cost = joint_result_without_storage[3]

picture23(combined_data, '联合', storage_capacity.X)

print('联合')
print(joint_avg_cost)
print()





(model,
 storage_charge, storage_discharge,
 storage_power_charge, storage_power_discharge,
 storage_energy, purchase, curtailment,
 storage_capacity, storage_power) = OptModel("energy_optimization", combined_data, 'A')


# 分析各园区配置储能后的经济性
park_A_result_with_storage = analyze_with_storage('A园区', 'A园区负荷(kW)', 'A园区光伏出力(kW)', 1, 0.4)


combined_data['A园区购电量(kW)'] = park_A_result_with_storage[0]
combined_data['A园区弃电量(kW)'] = park_A_result_with_storage[1]
combined_data['A园区_SOC(kWh)']= park_A_result_with_storage[4]
combined_data['A园区充放电(kWh)'] = combined_data[['A园区放电(kWh)', 'A园区充电(kWh)']].sum(axis=1)

joint_purchase = park_A_result_with_storage[0].sum()
joint_excess_generation = park_A_result_with_storage[1].sum()
joint_total_cost = park_A_result_with_storage[2]
joint_avg_cost = park_A_result_with_storage[3]

picture23(combined_data, 'A', storage_capacity.X)

print('A')
print(joint_avg_cost)
print()






(model,
 storage_charge, storage_discharge,
 storage_power_charge, storage_power_discharge,
 storage_energy, purchase, curtailment,
 storage_capacity, storage_power) = OptModel("energy_optimization", combined_data, 'B')


park_B_result_with_storage = analyze_with_storage('B园区', 'B园区负荷(kW)', 'B园区风电出力(kW)', 1, 0.5)

combined_data['B园区购电量(kW)'] = park_B_result_with_storage[0]
combined_data['B园区弃电量(kW)'] = park_B_result_with_storage[1]
combined_data['B园区_SOC(kWh)']= park_B_result_with_storage[4]
combined_data['B园区充放电(kWh)'] = combined_data[['B园区放电(kWh)', 'B园区充电(kWh)']].sum(axis=1)

joint_purchase = park_B_result_with_storage[0].sum()
joint_excess_generation = park_B_result_with_storage[1].sum()
joint_total_cost = park_B_result_with_storage[2]
joint_avg_cost = park_B_result_with_storage[3]

picture23(combined_data, 'B', storage_capacity.X)

print('B')
print(joint_avg_cost)
print()




(model,
 storage_charge, storage_discharge,
 storage_power_charge, storage_power_discharge,
 storage_energy, purchase, curtailment,
 storage_capacity, storage_power) = OptModel("energy_optimization", combined_data, 'C')

park_C_result_with_storage = analyze_with_storage('C园区', 'C园区负荷(kW)', ['C园区光伏出力(kW)', 'C园区风电出力(kW)'], 1, [0.4, 0.5])

combined_data['C园区购电量(kW)'] = park_C_result_with_storage[0]
combined_data['C园区弃电量(kW)'] = park_C_result_with_storage[1]
combined_data['C园区_SOC(kWh)']= park_C_result_with_storage[4]
combined_data['C园区充放电(kWh)'] = combined_data[['C园区放电(kWh)', 'C园区充电(kWh)']].sum(axis=1)

joint_purchase = park_C_result_with_storage[0].sum()
joint_excess_generation = park_C_result_with_storage[1].sum()
joint_total_cost = park_C_result_with_storage[2]
joint_avg_cost = park_C_result_with_storage[3]

picture23(combined_data, 'C', storage_capacity.X)

print('C')
print(joint_avg_cost)
print()

