# -*- coding: utf-8 -*-
"""
Created on Sun May 25 07:32:11 2024

@author: Gregory_Guo
"""


import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import os

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

efficiency_charge = 0.95  # 充电效率
efficiency_discharge = 0.95  # 放电效率
purchase_cost = 1  # 购电成本
pv_cost = 0.4  # 光伏发电成本
wind_cost = 0.5  # 风电发电成本
# 储能系统的成本参数
power_cost_per_kw = 800  # 元/kW
energy_cost_per_kwh = 1800  # 元/kWh


for initial_storage_energy in [0.1, 0.9]:
    
    # initial_storage_energy = 0.9  # 初始储能能量%
    
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
        
        Pv_used = model.addVars(time_steps, lb = 0, vtype=GRB.CONTINUOUS, name="Pv_used")
        Pw_used = model.addVars(time_steps, lb = 0, vtype=GRB.CONTINUOUS, name="Pw_used")
        Pg_used = model.addVars(time_steps, lb = 0, vtype=GRB.CONTINUOUS, name="Pg_used")
    
        # 新增的储能容量和储能功率决策变量
        storage_capacity = model.addVar(vtype=GRB.CONTINUOUS, name="storage_capacity")
        storage_power = model.addVar(vtype=GRB.CONTINUOUS, name="storage_power")
        
        # model.addConstr(storage_capacity == 100)
        # model.addConstr(storage_power == 50)
    
            
        # 约束条件
        for t in range(time_steps):
            load = combined_data.at[t, f'{park}园区负荷(kW)']
            pv_gen = combined_data.at[t, f'{park}园区光伏出力(kW)']
            wind_gen = combined_data.at[t, f'{park}园区风电出力(kW)']
            
            # 功率平衡约束
            model.addConstr(storage_power_charge[t] == Pv_used[t] + Pw_used[t] + Pg_used[t] + storage_power_discharge[t] - load)
    
            model.addConstr(Pv_used[t] <= pv_gen)
            model.addConstr(Pw_used[t] <= wind_gen)
        
            # 储能不能同时充电和放电
            model.addConstr(storage_charge[t] + storage_discharge[t] <= 1)
            
            # 储能充放电功率限制
            model.addConstr(storage_power_charge[t] <= storage_charge[t] * storage_power)
            model.addConstr(storage_power_discharge[t] <= storage_discharge[t] * storage_power)
            
            # 储能能量约束
            model.addConstr(storage_energy[t] <= 0.9 * storage_capacity)
            model.addConstr(storage_energy[t] >= 0.1 * storage_capacity)
            
            # 储能状态约束
            if t == 0:
                model.addConstr(storage_energy[t] == initial_storage_energy*storage_capacity + storage_power_charge[t] * efficiency_charge - storage_power_discharge[t] / efficiency_discharge)
            else:
                model.addConstr(storage_energy[t] == storage_energy[t-1] + storage_power_charge[t] * efficiency_charge - storage_power_discharge[t] / efficiency_discharge)
    
        
        # 初始和终端储能状态约束
        model.addConstr(storage_energy[0] == initial_storage_energy*storage_capacity)
        
        
        
        # model.addConstr(storage_energy[time_steps - 1] == initial_storage_energy*storage_capacity)
        
        # 目标函数：最小化总成本（包括购电成本、发电成本和储能系统的投资成本）
        total_cost = gp.quicksum(Pg_used[t] * purchase_cost+
                                 Pv_used[t] * pv_cost +
                                 Pw_used[t] * wind_cost
                                  for t in range(time_steps))
        total_investment_cost = (storage_power * power_cost_per_kw + storage_capacity * energy_cost_per_kwh)/365*0.1295
        model.setObjective(total_cost + total_investment_cost, GRB.MINIMIZE)
        
        # 求解模型
        model.optimize()
        
        return (model, storage_charge, storage_discharge,
                storage_power_charge, storage_power_discharge,
                storage_energy, storage_capacity, storage_power,
                Pv_used, Pw_used, Pg_used)
    
    
    def analyze_with_storage(data, park, load_col, gen_col, purchase_cost, generation_cost, OptData):
        """
        
        """
        
        (model, storage_charge, storage_discharge,
                storage_power_charge, storage_power_discharge,
                storage_energy, storage_capacity, storage_power,
                Pv_used, Pw_used, Pg_used) = OptData
    
        # 运行储能策略
        for i in range(0, len(combined_data)):
    
            data.at[i, f'{park}园区_SOC(kWh)'] = storage_energy[i].X
            data.at[i, f'{park}园区购电量(kW)'] = Pg_used[i].X
            data.at[i, f'{park}光伏用电量(kW)'] = Pv_used[i].X
            data.at[i, f'{park}风电用电量(kW)'] = Pw_used[i].X
            data.at[i, f'{park}放电(kWh)'] = -storage_power_discharge[i].X
            data.at[i, f'{park}充电(kWh)'] = storage_power_charge[i].X
            
        data[f'{park}园区充放电(kWh)'] = data[[f'{park}放电(kWh)', f'{park}充电(kWh)']].sum(axis=1)
    
            
        # 经济性分析
        load = data[load_col]
        
            # 分开处理光伏和风电发电
        # pv_generation = data[gen_col[0]]
        # wind_generation = data[gen_col[1]]
        pv_cost = generation_cost[0]
        wind_cost = generation_cost[1]
        # power_cost = generation_cost[2]
        # energy_cost = generation_cost[3]
    
        data[f'{park}园区弃光(kW)'] = data[f'{park}园区光伏出力(kW)'] - data[f'{park}光伏用电量(kW)'] 
        data[f'{park}园区弃风(kW)'] = data[f'{park}园区风电出力(kW)'] - data[f'{park}风电用电量(kW)']
        data[f'{park}园区弃电量(kW)'] = data[[f'{park}园区弃风(kW)', f'{park}园区弃光(kW)']].sum(axis=1)
    
        total_purchase_cost = (data[f'{park}园区购电量(kW)'] * purchase_cost).sum()
        total_generation_cost = (data[f'{park}光伏用电量(kW)']* pv_cost +
                                 data[f'{park}风电用电量(kW)']* wind_cost).sum()
        total_storage_cost = (storage_power.X * power_cost_per_kw + storage_capacity.X * energy_cost_per_kwh)/365*0.1295
        
        total_cost = total_purchase_cost + total_generation_cost + total_storage_cost
        avg_cost = total_cost / load.sum()
        
        return (total_cost, avg_cost, data,
                data[f'{park}园区_SOC(kWh)'],
                data[f'{park}园区弃风(kW)'],
                data[f'{park}园区弃光(kW)'],
                data[f'{park}园区弃电量(kW)'],
                data[f'{park}放电(kWh)'],
                data[f'{park}充电(kWh)'],
                data[f'{park}园区购电量(kW)'],
                data[f'{park}光伏用电量(kW)'],
                data[f'{park}风电用电量(kW)'])
    
    
    def picture23(combined_data, park, storage_capacity):
        """
        
        """
        
        
        # 创建子图
        fig, axes = plt.subplots(1, 1, figsize=(15, 0.72*10), sharex=True)
        
        # 定义柱状图的宽度和透明度
        bar_width = 0.7
        alpha = 0.5
        
        # 字体设置
        title_fontsize = 26
        label_fontsize = 24
        legend_fontsize = 25
        tick_labelsize = 24
        
        # 绘制园区C的数据
        axes.bar(combined_data['时间（h）'], combined_data[f'{park}园区光伏出力(kW)'], width=bar_width, label='P_V', color='lightpink', alpha=alpha)
        axes.bar(combined_data['时间（h）'], combined_data[f'{park}园区风电出力(kW)'], width=bar_width, bottom= combined_data[f'{park}园区光伏出力(kW)'], label='P_W', color='skyblue', alpha=alpha)
        bars_charge = axes.bar(combined_data['时间（h）'], combined_data[f'{park}园区充放电(kWh)'], width=bar_width, bottom= combined_data[f'{park}园区负荷(kW)'], label='P_C', color='purple', alpha=1)
        axes.bar(combined_data['时间（h）'], combined_data[f'{park}园区负荷(kW)'], width=bar_width, label='P_L', edgecolor='black', facecolor='none', linewidth=2, linestyle = '--')
       
        axes.bar(combined_data['时间（h）'], combined_data[f'{park}园区购电量(kW)'], width=bar_width, bottom= combined_data[f'{park}园区光伏出力(kW)']+combined_data[f'{park}园区风电出力(kW)'], label='P_G', color='orange', alpha=alpha)
        axes.bar(combined_data['时间（h）'], combined_data[f'{park}园区弃电量(kW)'], width=bar_width, bottom = -combined_data[f'{park}园区弃电量(kW)'], label='P_X', color='gray', alpha=alpha)
        
        if park == '联合':
            park = 'Joint'
            axes.set_title(f'Park {park}', fontsize=title_fontsize)
            park = '联合'
        else:
            axes.set_title(f'Park {park}', fontsize=title_fontsize)
        
        axes.set_xlabel('Time (h)', fontsize=label_fontsize)
        axes.set_ylabel('Power (kW)', fontsize=label_fontsize)
        # axes.legend(loc='upper left', fontsize=legend_fontsize)
        axes.tick_params(axis='both', which='major', labelsize=tick_labelsize)  # 调整刻度标签字体大小
    
        axes.legend(loc='upper center', fontsize=legend_fontsize, bbox_to_anchor=(0.5, 1.25), ncol=6)
    
        # 添加阴影效果
        for bar in bars_charge:
            bar.set_edgecolor('grey')
            bar.set_linewidth(1)
            bar.set_alpha(0.7)
        
        # 创建次坐标轴
        ax2 = axes.twinx()
        ax2.plot(combined_data['时间（h）'], combined_data[f'{park}园区_SOC(kWh)']/storage_capacity, label='SOC', color='green', marker='o')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('SOC', fontsize=label_fontsize)
        ax2.tick_params(axis='y', which='major', labelsize=tick_labelsize)  # 调整刻度标签字体大小
    
        ax2.legend(loc='upper right', fontsize=legend_fontsize)
    
        
        # 调整布局
        plt.tight_layout()
        plt.show()
    
    
    
    def Write(OptResults, EcoResults, fileName):
        """
        
        """
    
        results = {
        'storage_capacity': OptResults[6].X,
        'storage_power': OptResults[7].X,
        'total_cost': EcoResults[0],
        'avg_cost': EcoResults[1],
        'PX_pv': EcoResults[5],
        'PX_wind': EcoResults[4],
        'PX_total': EcoResults[6],
        'PC_discharge': EcoResults[7],
        'PC_charge': EcoResults[8],
        'PG': EcoResults[9],
        'PV': EcoResults[10],
        'PW': EcoResults[11],
        }
    
        df = pd.DataFrame(results)
        
        # 指定导出文件夹路径和文件名
        output_folder = r'C:\Users\29639\Desktop\2024电工杯\dataResults'  # 请将此路径替换为你需要的文件夹路径
        output_filename = fileName
        
        # 创建文件夹（如果不存在）
        os.makedirs(output_folder, exist_ok=True)
        
        # 生成完整的文件路径
        output_file = os.path.join(output_folder, output_filename)
        
        # 将DataFrame写入Excel文件
        df.to_excel(output_file, index=False)
    
    
    JointOpt = OptModel("energy_optimization", combined_data, '联合')
    joint_result_with_storage = analyze_with_storage(combined_data,
        "联合",
        '联合园区负荷(kW)', 
        ['联合园区光伏出力(kW)', '联合园区风电出力(kW)'], 
        purchase_cost, 
        [pv_cost, wind_cost, power_cost_per_kw, energy_cost_per_kwh], JointOpt
    )
    
    
    AOpt  = OptModel("energy_optimization", combined_data, 'A')
    park_A_result_with_storage = analyze_with_storage(combined_data,
        'A',
        'A园区负荷(kW)',
        ['A园区光伏出力(kW)','A园区风电出力(kW)'],
        purchase_cost, 
        [pv_cost, wind_cost, power_cost_per_kw, energy_cost_per_kwh], AOpt
    )
    
    
    BOpt  = OptModel("energy_optimization", combined_data, 'B')
    park_B_result_with_storage = analyze_with_storage(combined_data,
        'B',
        'B园区负荷(kW)',
        ['B园区光伏出力(kW)','B园区风电出力(kW)'],
        purchase_cost, 
        [pv_cost, wind_cost, power_cost_per_kw, energy_cost_per_kwh], BOpt
    )
    
    
    COpt  = OptModel("energy_optimization", combined_data, 'C')
    park_C_result_with_storage = analyze_with_storage(combined_data,
        'C',
        'C园区负荷(kW)',
        ['C园区光伏出力(kW)','C园区风电出力(kW)'],
        purchase_cost, 
        [pv_cost, wind_cost, power_cost_per_kw, energy_cost_per_kwh], COpt
    )
    
    
    picture23(joint_result_with_storage[2], '联合', JointOpt[6].X)
    
    picture23(park_A_result_with_storage[2], 'A', AOpt[6].X)
    
    picture23(park_B_result_with_storage[2], 'B', BOpt[6].X)
    
    picture23(park_C_result_with_storage[2], 'C', COpt[6].X)
    
    
    Write(AOpt, park_A_result_with_storage, f'AwithFlexible_{initial_storage_energy}.xlsx')
    Write(BOpt, park_B_result_with_storage, f'BwithFlexible_{initial_storage_energy}.xlsx')
    Write(COpt, park_C_result_with_storage, f'CwithFlexible_{initial_storage_energy}.xlsx')
    Write(JointOpt, joint_result_with_storage, f'JwithFlexible_{initial_storage_energy}.xlsx')

