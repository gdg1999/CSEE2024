# -*- coding: utf-8 -*-
"""
Created on Sun May 26 23:46:59 2024

@author: Gregory_Guo
"""


import gurobipy as gp
from gurobipy import GRB
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use( ['science',"grid","ieee"])

import os

########################################################
####################   Data Load   #####################
########################################################

# Load all necessary data for the first question analysis
load_data_path = r'C:/Users/29639/Desktop/2024电工杯/题目/1716500748329333/A题/附件1：各园区典型日负荷数据.xlsx'
generation_monthly_data_path = r'C:/Users/29639/Desktop/2024电工杯/题目/1716500748329333/A题/附件3：12个月各园区典型日风光发电数据.xlsx'

# Load the load data
load_data = pd.read_excel(load_data_path)
load_data.columns = ['时间（h）', 'A园区负荷(kW)', 'B园区负荷(kW)', 'C园区负荷(kW)']


# Load the wind and solar generation data
generation_monthly_data = pd.read_excel(generation_monthly_data_path, header=2, sheet_name=0)


# 遍历所有工作表
for sheet_name in pd.ExcelFile(generation_monthly_data_path).sheet_names:
    # 读取当前工作表的数据
    generation_monthly_data = pd.read_excel(generation_monthly_data_path, header=2, sheet_name=sheet_name)

    generation_monthly_data.columns = ['时间（h）', f'{sheet_name}A园区光伏出力（p.u.）', f'{sheet_name}B园区风电出力（p.u.）', f'{sheet_name}C园区光伏出力（p.u.）', f'{sheet_name}C园区风电出力（p.u.）']
    
    
    # Combine load and generation data for each park
    load_data = pd.merge(load_data, generation_monthly_data, on='时间（h）')
    
    load_data[f'{sheet_name}A园区风电出力（p.u.）'] = 0
    load_data[f'{sheet_name}B园区光伏出力（p.u.）'] = 0

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

# 汇总联合园区的负荷和发电数据
# combined_data['联合园区负荷(kW)'] = combined_data[['A园区负荷(kW)', 'B园区负荷(kW)', 'C园区负荷(kW)']].sum(axis=1)
# combined_data['联合园区光伏出力（p.u.）'] = combined_data[['A园区光伏出力（p.u.）', 'C园区光伏出力（p.u.）']].sum(axis=1)
# combined_data['联合园区风电出力（p.u.）'] = combined_data[['B园区风电出力（p.u.）', 'C园区风电出力（p.u.）']].sum(axis=1)



# 初始储能状态
initial_storage_energy = 0.1  # 初始储能能量%
efficiency_charge = 0.95  # 充电效率
efficiency_discharge = 0.95  # 放电效率
purchase_cost = 1  # 购电成本
pv_cost = 0.4  # 光伏发电成本
wind_cost = 0.5  # 风电发电成本

# 储能系统的成本参数
power_cost_per_kw = 80  # 元/kW
energy_cost_per_kwh = 180  # 元/kWh

# 风光系统的成本参数
wind_cost_per_kw = 3000  # 风电容量成本，元/kW
solar_cost_per_kw = 2500  # 光伏容量成本，元/kW

########################################################
####################   Opt Model   #####################
########################################################
# (modelCase, combined_data, park) = ("energy_optimization", combined_data, 'C')
def OptModel(modelCase, combined_data, park):

    # 创建模型
    model = gp.Model(modelCase)
    
    # 时间步长
    time_steps = combined_data.shape[0]
    # 月份
    month_num = 12
    
    # cost for renewable energy
    Gain = 500
    
    # 创建变量
    storage_charge = model.addVars(time_steps, month_num, vtype=GRB.BINARY, name="storage_charge")
    storage_discharge = model.addVars(time_steps, month_num, vtype=GRB.BINARY, name="storage_discharge")
    
    storage_energy = model.addVars(time_steps, month_num, vtype=GRB.CONTINUOUS, name="storage_energy")
    
    storage_power_charge = model.addVars(time_steps, month_num, vtype=GRB.CONTINUOUS, name="storage_power_charge")
    storage_power_discharge = model.addVars(time_steps, month_num, vtype=GRB.CONTINUOUS, name="storage_power_discharge")
    
    Pv_used = model.addVars(time_steps, month_num, lb = 0, vtype=GRB.CONTINUOUS, name="Pv_used")
    Pw_used = model.addVars(time_steps, month_num, lb = 0, vtype=GRB.CONTINUOUS, name="Pw_used")
    Pg_used = model.addVars(time_steps, month_num, lb = 0, vtype=GRB.CONTINUOUS, name="Pg_used")

    # 新增的储能容量和储能功率决策变量
    storage_capacity = model.addVar(vtype=GRB.CONTINUOUS, name="storage_capacity")
    storage_power = model.addVar(vtype=GRB.CONTINUOUS, name="storage_power")
    
    # 风光容量
    wind_capacity = model.addVar(vtype=GRB.CONTINUOUS, name="wind_capacity")
    solar_capacity = model.addVar(vtype=GRB.CONTINUOUS, name="solar_capacity")
    
    Cost_energy = model.addVars(time_steps, month_num, lb = 0, vtype=GRB.CONTINUOUS, name="Cost_energy")
        
    # 约束条件
    for m in range(month_num):
        for t in range(time_steps):
            load = combined_data.at[t, f'{park}园区负荷(kW)'] * 1.5
            
            pv_gen = combined_data.at[t, f'M{m+1}{park}园区光伏出力（p.u.）'] * solar_capacity
            wind_gen = combined_data.at[t, f'M{m+1}{park}园区风电出力（p.u.）'] * wind_capacity
            
            model.addConstr(Pv_used[t,m] <= pv_gen)
            model.addConstr(Pw_used[t,m] <= wind_gen)
            
            # 功率平衡约束
            model.addConstr(storage_power_charge[t,m] == Pv_used[t,m] + Pw_used[t,m] + Pg_used[t,m] + storage_power_discharge[t,m] - load)
       
            # 储能不能同时充电和放电
            model.addConstr(storage_charge[t,m] + storage_discharge[t,m] <= 1)
            
            # 储能充放电功率限制
            model.addConstr(storage_power_charge[t,m] <= storage_charge[t,m] * storage_power)
            model.addConstr(storage_power_discharge[t,m] <= storage_discharge[t,m] * storage_power)
            
            # 储能能量约束
            model.addConstr(storage_energy[t,m] <= 0.9 * storage_capacity)
            model.addConstr(storage_energy[t,m] >= 0.1 * storage_capacity)
            
            # 储能状态转移约束
            if t == 0 & m == 0:
                model.addConstr(storage_energy[t,m] == initial_storage_energy*storage_capacity + storage_power_charge[t,m] * efficiency_charge - storage_power_discharge[t,m] / efficiency_discharge)
            elif t == 0 & m != 0:
                model.addConstr(storage_energy[t,m] == storage_energy[time_steps-1,m-1] + storage_power_charge[t,m] * efficiency_charge - storage_power_discharge[t,m] / efficiency_discharge)
            else:
                model.addConstr(storage_energy[t,m] == storage_energy[t-1,m] + storage_power_charge[t,m] * efficiency_charge - storage_power_discharge[t,m] / efficiency_discharge)
    
            # model.addConstr(Cost_energy[t,m] == Gain*(load - storage_power_discharge[t,m] - Pv_used[t,m] - Pw_used[t,m]))
            model.addConstr(Cost_energy[t,m] == Gain*(load - Pv_used[t,m] - Pw_used[t,m]))
    
        # 初始和终端储能状态约束
        model.addConstr(storage_energy[0,0] == initial_storage_energy*storage_capacity)
        model.addConstr(storage_energy[time_steps - 1, month_num-1] == initial_storage_energy*storage_capacity)
        
    # 目标函数：最小化总成本（包括购电成本、发电成本和储能系统的投资成本）
    total_cost = gp.quicksum(Pg_used[t,m] * combined_data.at[t, '电价']+
                             Pv_used[t,m] * pv_cost +
                             Pw_used[t,m] * wind_cost
                              for t in range(time_steps) for m in range(month_num))
    
    total_investment_cost = (storage_power * power_cost_per_kw + storage_capacity * energy_cost_per_kwh)/10
    
    wind_investment_cost = wind_capacity * wind_cost_per_kw /5
    
    solar_investment_cost = solar_capacity * solar_cost_per_kw /5
    
    Total_Construction_Cost = total_investment_cost + wind_investment_cost + solar_investment_cost
    
    Original_cost = gp.quicksum(1.5* combined_data.at[t, '电价']*combined_data.at[t, f'{park}园区负荷(kW)'] for t in range(time_steps))
    
    # pay back constraint
    model.addConstr(30*12*Original_cost - 30*total_cost >= Total_Construction_Cost, name="sum_constraint")
    
    Cg = gp.quicksum(Cost_energy[t,m] for t in range(time_steps) for m in range(month_num))

    # model.addConstr(wind_capacity == 1000)
    # model.addConstr(solar_capacity == 1000)
    
    model.setObjective(Cg + total_cost + Total_Construction_Cost, GRB.MINIMIZE)
    
    model.setParam('MIPGap', 0.05)
    model.setParam('Timelimit', 300)
    
    # 求解模型
    model.optimize()
    
    # model.computeIIS()
    # # #
    # model.write("abc11111.ilp")
    
    return (model, storage_charge, storage_discharge,
            storage_power_charge, storage_power_discharge,
            storage_energy, storage_capacity, storage_power,
            Pv_used, Pw_used, Pg_used,
            wind_capacity, solar_capacity)


def analyze_with_storage(data, park, load_col, gen_col, purchase_cost, generation_cost, OptData):
    """
    
    """
    
    (model, storage_charge, storage_discharge,
            storage_power_charge, storage_power_discharge,
            storage_energy, storage_capacity, storage_power,
            Pv_used, Pw_used, Pg_used,
            wind_capacity, solar_capacity) = OptData

    # 运行储能策略
    
    for mm in range(0,12):
        for i in range(0, len(combined_data)):
            m = mm+1
            data.at[i, f'M{m}{park}园区_SOC(kWh)'] = storage_energy[i,mm].X
            data.at[i, f'M{m}{park}园区购电量(kW)'] = Pg_used[i,mm].X
            data.at[i, f'M{m}{park}光伏用电量(kW)'] = Pv_used[i,mm].X
            data.at[i, f'M{m}{park}风电用电量(kW)'] = Pw_used[i,mm].X
            data.at[i, f'M{m}{park}放电(kWh)'] = -storage_power_discharge[i,mm].X
            data.at[i, f'M{m}{park}充电(kWh)'] = storage_power_charge[i,mm].X
            
        data[f'M{m}{park}园区充放电(kWh)'] = data[[f'M{m}{park}放电(kWh)', f'M{m}{park}充电(kWh)']].sum(axis=1)

        
    # 经济性分析
    load = data[load_col]
    
    # 分开处理光伏和风电发电

    pv_cost = generation_cost[0]
    wind_cost = generation_cost[1]


    Data_dict = dict()

    for mm in range(12):
        m = mm + 1
        data[f'M{m}{park}园区弃光(kW)'] = data[f'M{m}{park}园区光伏出力（p.u.）']*solar_capacity.X - data[f'M{m}{park}光伏用电量(kW)']
        data[f'M{m}{park}园区弃风(kW)'] = data[f'M{m}{park}园区风电出力（p.u.）']*wind_capacity.X - data[f'M{m}{park}风电用电量(kW)']
        data[f'M{m}{park}园区弃电量(kW)'] = data[[f'M{m}{park}园区弃风(kW)', f'M{m}{park}园区弃光(kW)']].sum(axis=1)
    
        total_purchase_cost = (data.at[i, f'M{m}{park}园区购电量(kW)'] * data.at[i, '电价']).sum()
        total_generation_cost = (data.at[i, f'M{m}{park}光伏用电量(kW)']* pv_cost +
                                 data.at[i, f'M{m}{park}风电用电量(kW)']* wind_cost).sum()
        total_storage_cost = (storage_power.X * power_cost_per_kw + storage_capacity.X * energy_cost_per_kwh)/365*0.1295
        
        total_cost = total_purchase_cost + total_generation_cost + total_storage_cost
        avg_cost = total_cost / load.sum()
        
        Data_dict[f'{m}'] = (total_cost, avg_cost, data,
                             data[f'M{m}{park}园区_SOC(kWh)'],
                             data[f'M{m}{park}园区弃风(kW)'],
                             data[f'M{m}{park}园区弃光(kW)'],
                             data[f'M{m}{park}园区弃电量(kW)'],
                             data[f'M{m}{park}放电(kWh)'],
                             data[f'M{m}{park}充电(kWh)'],
                             data[f'M{m}{park}园区购电量(kW)'],
                             data[f'M{m}{park}光伏用电量(kW)'],
                             data[f'M{m}{park}风电用电量(kW)'])
    
    return Data_dict


def picture23(combined_data, m, park, storage_capacity, wind_capacity, solar_capacity):
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
    axes.bar(combined_data['时间（h）'], solar_capacity*combined_data[f'M{m}{park}园区光伏出力（p.u.）'], width=bar_width, label='P_V', color='lightpink', alpha=alpha)
    axes.bar(combined_data['时间（h）'], wind_capacity*combined_data[f'M{m}{park}园区风电出力（p.u.）'], width=bar_width, bottom= solar_capacity*combined_data[f'M{m}{park}园区光伏出力（p.u.）'], label='P_W', color='skyblue', alpha=alpha)
    bars_charge = axes.bar(combined_data['时间（h）'], combined_data[f'M{m}{park}园区充放电(kWh)'], width=bar_width, bottom= 1.5*combined_data[f'{park}园区负荷(kW)'], label='P_C', color='purple', alpha=1)
    axes.bar(combined_data['时间（h）'], 1.5*combined_data[f'{park}园区负荷(kW)'], width=bar_width, label='P_L', edgecolor='black', facecolor='none', linewidth=2, linestyle = '--')
   
    axes.bar(combined_data['时间（h）'], combined_data[f'M{m}{park}园区购电量(kW)'], width=bar_width, bottom= solar_capacity*combined_data[f'M{m}{park}园区光伏出力（p.u.）']+wind_capacity*combined_data[f'M{m}{park}园区风电出力（p.u.）'], label='P_G', color='orange', alpha=alpha)
    axes.bar(combined_data['时间（h）'], combined_data[f'M{m}{park}园区弃电量(kW)'], width=bar_width, bottom = -combined_data[f'M{m}{park}园区弃电量(kW)'], label='P_X', color='gray', alpha=alpha)
    
    if park == '联合':
        park = 'Joint'
        axes.set_title(f'M{m}_Park {park}', fontsize=title_fontsize)
        park = '联合'
    else:
        axes.set_title(f'M{m}_Park {park}', fontsize=title_fontsize)
    
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
    ax2.plot(combined_data['时间（h）'], combined_data[f'M{m}{park}园区_SOC(kWh)']/storage_capacity, label='SOC', color='green', marker='o')
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


# JointOpt = OptModel("energy_optimization", combined_data, '联合')
# joint_result_with_storage = analyze_with_storage(combined_data,
#     "联合",
#     '联合园区负荷(kW)', 
#     ['联合园区光伏出力(kW)', '联合园区风电出力(kW)'], 
#     purchase_cost, 
#     [pv_cost, wind_cost, power_cost_per_kw, energy_cost_per_kwh], JointOpt
# )


# AOpt  = OptModel("energy_optimization", combined_data, 'A')
# park_A_result_with_storage = analyze_with_storage(combined_data,
#     'A',
#     'A园区负荷(kW)',
#     ['A园区光伏出力（p.u.）','A园区风电出力（p.u.）'],
#     purchase_cost, 
#     [pv_cost, wind_cost, power_cost_per_kw, energy_cost_per_kwh], AOpt
# )


# BOpt  = OptModel("energy_optimization", combined_data, 'B')
# park_B_result_with_storage = analyze_with_storage(combined_data,
#     'B',
#     'B园区负荷(kW)',
#     ['B园区光伏出力(kW)','B园区风电出力(kW)'],
#     purchase_cost, 
#     [pv_cost, wind_cost, power_cost_per_kw, energy_cost_per_kwh], BOpt
# )


COpt  = OptModel("energy_optimization", combined_data, 'C')
park_C_Dict = analyze_with_storage(combined_data,
    'C',
    'C园区负荷(kW)',
    ['C园区光伏出力(kW)','C园区风电出力(kW)'],
    purchase_cost, 
    [pv_cost, wind_cost, power_cost_per_kw, energy_cost_per_kwh], COpt
)


# picture23(joint_result_with_storage[2], '联合', JointOpt[6].X, JointOpt[11].X, JointOpt[12].X)

# picture23(park_A_result_with_storage[2], 'A', AOpt[6].X, AOpt[11].X, AOpt[12].X)

# picture23(park_B_result_with_storage[2], 'B', BOpt[6].X, BOpt[11].X, BOpt[12].X)


for m in range(12):
    # print(park_C_Dict[f'{m+1}'][2]['M1C园区充放电(kWh)'])
    picture23(park_C_Dict[f'{m+1}'][2], m+1, 'C', COpt[6].X, COpt[11].X, COpt[12].X)


# Write(AOpt, park_A_result_with_storage, f'AwithFlexible_{initial_storage_energy}.xlsx')
# Write(BOpt, park_B_result_with_storage, f'BwithFlexible_{initial_storage_energy}.xlsx')
# Write(COpt, park_C_result_with_storage, f'CwithFlexible_{initial_storage_energy}.xlsx')
# Write(JointOpt, joint_result_with_storage, f'JwithFlexible_{initial_storage_energy}.xlsx')