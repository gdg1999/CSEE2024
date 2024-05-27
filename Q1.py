# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:19:32 2024

@author: Gregory_Guo
"""

import pandas as pd
import numpy as np
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
combined_data['时间（h）'] = pd.to_datetime(combined_data['时间（h）'], format='%H:%M:%S').dt.hour

# 汇总联合园区的负荷和发电数据
combined_data['联合园区负荷(kW)'] = combined_data[['园区A负荷(kW)', '园区B负荷(kW)', '园区C负荷(kW)']].sum(axis=1)
combined_data['联合园区光伏出力(kW)'] = combined_data[['园区A光伏出力(kW)', '园区C光伏出力(kW)']].sum(axis=1)
combined_data['联合园区风电出力(kW)'] = combined_data[['园区B风电出力(kW)', '园区C风电出力(kW)']].sum(axis=1)

########################################################
######################   Storage   #####################
########################################################

# 储能系统参数
storage_power = 50  # kW
storage_capacity = 100  # kWh
efficiency = 0.95  # 充放电效率
soc_min = 0.1 * storage_capacity  # 最小充电状态
soc_max = 0.9 * storage_capacity  # 最大充电状态
charge_efficiency =  efficiency

########################################################
#####################   Function   #####################
########################################################

def analyze_without_storage(load_col, gen_col, purchase_cost, generation_cost):
    """
    未配置储能的经济性分析函数
    """
    
    load = combined_data[load_col]

    if isinstance(gen_col, list):
        # 分开处理光伏和风电发电
        pv_generation = combined_data[gen_col[0]]
        wind_generation = combined_data[gen_col[1]]
        pv_cost = generation_cost[0]
        wind_cost = generation_cost[1]

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
        excess_pv_generation = pv_generation - used_pv_generation
        excess_wind_generation = wind_generation - used_wind_generation
        excess_generation = excess_pv_generation + excess_wind_generation

        # 实际用电成本
        total_generation_cost = (used_pv_generation * pv_cost).sum() + (used_wind_generation * wind_cost).sum()

    else:
        total_generation = combined_data[gen_col]
        used_generation = total_generation.copy()
        used_generation[used_generation > load] = load[used_generation > load]
        remaining_load = load - used_generation

        excess_generation = total_generation - used_generation
        excess_generation[excess_generation < 0] = 0

        total_used_generation = used_generation
        total_generation_cost = (total_used_generation * generation_cost).sum()
        
    net_load = load - total_used_generation
    purchase = net_load.copy()
    purchase[purchase < 0] = 0  # Set negative values to zero, as excess generation is not sold

    total_purchase_cost = (purchase * purchase_cost).sum()
    total_cost = total_purchase_cost + total_generation_cost
    avg_cost = total_cost / load.sum()

    return purchase, excess_generation, total_cost, avg_cost, total_used_generation.sum()



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

# def storage_operation(row, previous_soc, load_col, gen_col, generation_cost):
#     """
#     储能运行策略
#     """
    
#     load = row[load_col]
    
#     if isinstance(gen_col, list):
#         # 分开处理光伏和风电发电
#         pv_generation = combined_data[gen_col[0]]
#         wind_generation = combined_data[gen_col[1]]
#         pv_cost = generation_cost[0]
#         wind_cost = generation_cost[1]

#         total_generation = pv_generation + wind_generation
        
#         net_load = load - total_generation
        
#         if net_load > 0:
#             # 需要从储能放电
#             charge = 0
#             discharge = min(storage_power, net_load / efficiency, previous_soc - soc_min)
#             new_soc = previous_soc - discharge
#             net_load -= discharge * efficiency
        
#         else:
#             # 需要给储能充电
#             discharge = 0
#             charge = min(storage_power, -net_load * efficiency, soc_max - previous_soc)

#             remaining_net_load = net_load
            
#             # 先使用光伏发电充电
#             if pv_generation - load > 0:
#                 charge_pv = min(storage_power, -remaining_net_load * charge_efficiency, soc_max - previous_soc, pv_generation)
#                 new_soc = previous_soc + charge_pv
#                 remaining_net_load += charge_pv / charge_efficiency
#             else:
#                 charge_pv = 0
            
#             charge - wind_generation
            
#             # 如果光伏发电不足，再使用风电充电
#             if remaining_net_load < 0 and wind_generation > 0:
#                 charge_wind = min(storage_power, -remaining_net_load * charge_efficiency, soc_max - new_soc, wind_generation)
#                 new_soc += charge_wind
#                 remaining_net_load += charge_wind / charge_efficiency
#             else:
#                 charge_wind = 0
            
#             charge = charge_pv + charge_wind
            
#             new_soc = previous_soc + charge
#             net_load += charge / efficiency
        
        
#         # 优先使用光伏发电
#         used_pv_generation = pv_generation.copy()
#         used_pv_generation[used_pv_generation > load] = load[used_pv_generation > load]
#         remaining_load = load - used_pv_generation

#         # 使用风电发电补充
#         used_wind_generation = wind_generation.copy()
#         used_wind_generation[used_wind_generation > remaining_load] = remaining_load[used_wind_generation > remaining_load]
#         remaining_load -= used_wind_generation

#         # 总的实际使用发电量
#         total_used_generation = used_pv_generation + used_wind_generation

#         # 弃风弃光量
#         # excess_pv_generation = pv_generation - used_pv_generation
#         # excess_wind_generation = wind_generation - used_wind_generation
#         # excess_generation = excess_pv_generation + excess_wind_generation

#         # 实际用电成本
#         total_generation_cost = (used_pv_generation * pv_cost).sum() + (used_wind_generation * wind_cost).sum()

#     else:
#         total_generation = combined_data[gen_col]
#         used_generation = total_generation.copy()
#         used_generation[used_generation > load] = load[used_generation > load]
#         remaining_load = load - used_generation

#         # excess_generation = total_generation - used_generation
#         # excess_generation[excess_generation < 0] = 0

#         total_used_generation = used_generation
#         total_generation_cost = (total_used_generation * generation_cost).sum()
    
    


        
#     return new_soc, net_load, discharge, charge


# def storage_operation(row, previous_soc, load_col, pv_col, wind_col):
#     """
#     储能运行策略
#     """
    
#     load = row[load_col]
#     pv_generation = row[pv_col]
#     wind_generation = row[wind_col]
    
#     # 计算净负荷
#     net_load = load - pv_generation - wind_generation
    
#     if net_load > 0:
#         # 需要从储能放电
#         charge = 0
#         discharge = min(storage_power, net_load / discharge_efficiency, previous_soc - soc_min)
#         new_soc = previous_soc - discharge
#         net_load -= discharge * discharge_efficiency
#     else:
#         # 需要给储能充电
#         discharge = 0
#         remaining_net_load = net_load
        
#         # 先使用光伏发电充电
#         if pv_generation > 0:
#             charge_pv = min(storage_power, -remaining_net_load * charge_efficiency, soc_max - previous_soc, pv_generation)
#             new_soc = previous_soc + charge_pv
#             remaining_net_load += charge_pv / charge_efficiency
#         else:
#             charge_pv = 0
        
#         # 如果光伏发电不足，再使用风电充电
#         if remaining_net_load < 0 and wind_generation > 0:
#             charge_wind = min(storage_power, -remaining_net_load * charge_efficiency, soc_max - new_soc, wind_generation)
#             new_soc += charge_wind
#             remaining_net_load += charge_wind / charge_efficiency
#         else:
#             charge_wind = 0
        
#         charge = charge_pv + charge_wind
        
#     return new_soc, net_load, discharge, charge


def analyze_with_storage(park, load_col, gen_col, purchase_cost, generation_cost):
    """
    
    """
    
    # 初始化储能状态
    xInitial_storage = 50
    # combined_data.at[0, 'SOC (kWh)'] = 50  # 初始SOC设为50kWh
    
    # combined_data[f'{park}净负荷(kW)'] = 0  # 初始化净负荷列

    # 运行储能策略
    for i in range(0, len(combined_data)):
        
        if i == 0:
            previous_soc = xInitial_storage

        else:
            previous_soc = combined_data.at[i-1, f'园区{park}_SOC (kWh)']
        new_soc, net_load, discharge, charge = storage_operation(combined_data.iloc[i], previous_soc, load_col, gen_col)
        combined_data.at[i, f'园区{park}_SOC (kWh)'] = new_soc
        combined_data.at[i, f'{park}净负荷(kW)'] = net_load
        combined_data.at[i, f'{park}放电(kWh)'] = -discharge
        combined_data.at[i, f'{park}充电(kWh)'] = charge

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
    purchase = net_load
    purchase[purchase < 0] = 0  # Set negative values to zero, as excess generation is not sold
    excess_generation = total_generation - load - combined_data[f'{park}充电(kWh)']  # 弃电量要变化
    excess_generation[excess_generation < 0] = 0  # Set negative values to zero
    
    total_purchase_cost = (purchase * purchase_cost).sum()
    total_cost = total_purchase_cost + total_generation_cost + (storage_power * 800 + storage_capacity * 1800)/365*0.1295
    avg_cost = total_cost / load.sum()
    
    return (purchase, excess_generation,
                                total_cost, avg_cost,
                                combined_data[f'园区{park}_SOC (kWh)'],
                                combined_data[f'{park}净负荷(kW)'],
                                combined_data[f'{park}放电(kWh)'],
                                combined_data[f'{park}充电(kWh)'])


def print_results(park, result_without, result_with):
    """
    
    """
    
    print(f"{park} without storage:")
    print("  Purchase (kWh):", result_without[0])
    print("  Excess Generation (kWh):", result_without[1])
    print("  Total Cost (元):", result_without[2])
    print("  Average Cost (元/kWh):", result_without[3])
    
    print(f"{park} with storage:")
    print("  Purchase (kWh):", result_with[0])
    print("  Excess Generation (kWh):", result_with[1])
    print("  Total Cost (元):", result_with[2])
    print("  Average Cost (元/kWh):", result_with[3])
    
    improvement = result_without[2] - result_with[2]
    if improvement > 0:
        print(f"  Economic improvement: {improvement} 元 (total cost reduction)")
    else:
        print(f"  Economic degradation: {-improvement} 元 (total cost increase)")
    print()
    
    
def explain_results(park, result_without, result_with):
    """
    详细解释各园区的经济性变化原因
    """
    
    if result_with[2] < result_without[2]:
        print(f"{park} shows economic improvement with storage. The reasons are:")
        print('')
        if result_with[0] < result_without[0]:
            print("  - Reduced purchase from the grid due to better utilization of stored energy.")
            print('')
        if result_with[1] < result_without[1]:
            print("  - Reduced excess generation (curtailment) due to storage absorbing excess generation.")
            print('')
        print("  - The storage system has effectively balanced the load and generation, leading to overall cost savings.")
        print('')
    else:
        print(f"{park} shows economic degradation with storage. The reasons are:")
        print('')
        if result_with[0] > result_without[0]:
            print("  - Increased purchase from the grid due to inefficiencies in storage utilization or higher demand.")
            print('')
        if result_with[1] > result_without[1]:
            print("  - Increased excess generation (curtailment) due to suboptimal storage strategy.")
            print('')
        print("  - The cost of storage operation (charging and discharging losses) may have outweighed the benefits.")
        print('')


def picture11(combined_data):
    # 创建子图
    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    
    # 定义柱状图的宽度和透明度
    bar_width = 0.7
    alpha = 0.5
    
    # 字体设置
    title_fontsize = 26
    label_fontsize = 24
    legend_fontsize = 25
    tick_labelsize = 24
    
    # 绘制园区A的数据
    axes[0].bar(combined_data['时间（h）'], combined_data['园区A光伏出力(kW)'], width=bar_width, label='Park A PV Output (kW)', color='lightpink', alpha=alpha)
    axes[0].bar(combined_data['时间（h）'], combined_data['园区A负荷(kW)'], width=bar_width, label='Park A Load (kW)', edgecolor='black', facecolor='none', linewidth=2, linestyle = '--')
    axes[0].bar(combined_data['时间（h）'], combined_data['园区A购电量(kW)'], width=bar_width, bottom= combined_data['园区A光伏出力(kW)'], label='Park A Purchase (kW)', color='orange', alpha=alpha)
    axes[0].bar(combined_data['时间（h）'], combined_data['园区A弃电量(kW)'], width=bar_width, bottom = -combined_data['园区A弃电量(kW)'], label='Park A Excess (kW)', color='gray', alpha=alpha)
    
    axes[0].set_title('Park A', fontsize=title_fontsize)
    axes[0].set_ylabel('Power (kW)', fontsize=label_fontsize)
    # axes[0].legend(loc='upper right', fontsize=legend_fontsize)
    # axes[0].legend(loc='upper center', fontsize=legend_fontsize, bbox_to_anchor=(0.5, 1.15), ncol=4)
    axes[0].tick_params(axis='both', which='major', labelsize=tick_labelsize)  # 调整刻度标签字体大小

    
    # 绘制园区B的数据
    axes[1].bar(combined_data['时间（h）'], combined_data['园区B风电出力(kW)'], width=bar_width, label='Park B Wind Output (kW)', color='skyblue', alpha=alpha)
    axes[1].bar(combined_data['时间（h）'], combined_data['园区B负荷(kW)'], width=bar_width, label='Park B Load (kW)', edgecolor='black', facecolor='none', linewidth=2, linestyle = '--')
    axes[1].bar(combined_data['时间（h）'], combined_data['园区B购电量(kW)'], width=bar_width, bottom= combined_data['园区B风电出力(kW)'], label='Park B Purchase (kW)', color='orange', alpha=alpha)
    axes[1].bar(combined_data['时间（h）'], combined_data['园区B弃电量(kW)'], width=bar_width, bottom = -combined_data['园区B弃电量(kW)'], label='Park B Excess (kW)', color='gray', alpha=alpha)
    
    axes[1].set_title('Park B', fontsize=title_fontsize)
    axes[1].set_ylabel('Power (kW)', fontsize=label_fontsize)
    # axes[1].legend(loc='upper right', fontsize=legend_fontsize)
    # axes[1].legend(loc='upper center', fontsize=legend_fontsize, bbox_to_anchor=(0.5, 1.15), ncol=4)
    axes[1].tick_params(axis='both', which='major', labelsize=tick_labelsize)  # 调整刻度标签字体大小

    
    # 绘制园区C的数据
    axes[2].bar(combined_data['时间（h）'], combined_data['园区C光伏出力(kW)'], width=bar_width, label='P_V', color='lightpink', alpha=alpha)
    axes[2].bar(combined_data['时间（h）'], combined_data['园区C风电出力(kW)'], width=bar_width, bottom= combined_data['园区C光伏出力(kW)'], label='P_W', color='skyblue', alpha=alpha)
    axes[2].bar(combined_data['时间（h）'], combined_data['园区C负荷(kW)'], width=bar_width, label='P_L', edgecolor='black', facecolor='none', linewidth=2, linestyle = '--')
    axes[2].bar(combined_data['时间（h）'], combined_data['园区C购电量(kW)'], width=bar_width, bottom= combined_data['园区C光伏出力(kW)']+combined_data['园区C风电出力(kW)'], label='P_G', color='orange', alpha=alpha)
    axes[2].bar(combined_data['时间（h）'], combined_data['园区C弃电量(kW)'], width=bar_width, bottom = -combined_data['园区C弃电量(kW)'], label='P_X', color='gray', alpha=alpha)
    
    axes[2].set_title('Park C', fontsize=title_fontsize)
    axes[2].set_xlabel('Time (h)', fontsize=label_fontsize)
    axes[2].set_ylabel('Power (kW)', fontsize=label_fontsize)
    # axes[2].legend(loc='upper right', fontsize=legend_fontsize)
    axes[2].legend(loc='upper center', fontsize=legend_fontsize, bbox_to_anchor=(0.46, 3.82), ncol=5)
    axes[2].tick_params(axis='both', which='major', labelsize=tick_labelsize)  # 调整刻度标签字体大小

    
    # 调整布局
    plt.tight_layout()
    plt.show()

def picture12(combined_data):
    # 创建子图
    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    
    # 定义柱状图的宽度和透明度
    bar_width = 0.7
    alpha = 0.5
    
    # 字体设置
    title_fontsize = 26
    label_fontsize = 24
    legend_fontsize = 25
    tick_labelsize = 24
    
    # 绘制园区A的数据
    axes[0].bar(combined_data['时间（h）'], combined_data['园区A光伏出力(kW)'], width=bar_width, label='P_V', color='lightpink', alpha=alpha)
    bars_charge = axes[0].bar(combined_data['时间（h）'], combined_data['园区A_充放电(kWh)'], width=bar_width, bottom= combined_data['园区A负荷(kW)'], label='P_C', color='purple', alpha=1)
    axes[0].bar(combined_data['时间（h）'], combined_data['园区A负荷(kW)'], width=bar_width, label='P_L', edgecolor='black', facecolor='none', linewidth=2, linestyle = '--')
    
    axes[0].bar(combined_data['时间（h）'], combined_data['园区A购电量(kW)'], width=bar_width, bottom= combined_data['园区A光伏出力(kW)'], label='P_G', color='orange', alpha=alpha)
    axes[0].bar(combined_data['时间（h）'], combined_data['园区A弃电量(kW)'], width=bar_width, bottom = -combined_data['园区A弃电量(kW)'], label='P_X', color='gray', alpha=alpha)
    
    axes[0].set_title('Park A', fontsize=title_fontsize)
    axes[0].set_ylabel('Power (kW)', fontsize=label_fontsize)
    # axes[0].legend(loc='upper right', fontsize=legend_fontsize)
    axes[0].tick_params(axis='both', which='major', labelsize=tick_labelsize)  # 调整刻度标签字体大小

    # 添加阴影效果
    for bar in bars_charge:
        bar.set_edgecolor('grey')
        bar.set_linewidth(1)
        bar.set_alpha(0.7)
        
    # 创建次坐标轴
    ax0 = axes[0].twinx()
    ax0.plot(combined_data['时间（h）'], combined_data['园区A_SOC(kWh)']/storage_capacity, label='SOC', color='green', marker='o')
    ax0.set_ylim(0, 1)
    ax0.set_ylabel('SOC', fontsize=label_fontsize)
    ax0.tick_params(axis='y', which='major', labelsize=tick_labelsize)  # 调整刻度标签字体大小    
    ax0.legend(loc='upper right', fontsize=legend_fontsize)

    # 绘制园区B的数据
    axes[1].bar(combined_data['时间（h）'], combined_data['园区B风电出力(kW)'], width=bar_width, label='Park B Wind Output (kW)', color='skyblue', alpha=alpha)
    bars_charge = axes[1].bar(combined_data['时间（h）'], combined_data['园区B_充放电(kWh)'], width=bar_width, bottom= combined_data['园区B负荷(kW)'], label='Park B Charge  (kW)', color='purple', alpha=1)
    axes[1].bar(combined_data['时间（h）'], combined_data['园区B负荷(kW)'], width=bar_width, label='Park B Load (kW)', edgecolor='black', facecolor='none', linewidth=2, linestyle = '--')
    
    axes[1].bar(combined_data['时间（h）'], combined_data['园区B购电量(kW)'], width=bar_width, bottom= combined_data['园区B风电出力(kW)'], label='Park B Purchase (kW)', color='orange', alpha=alpha)
    axes[1].bar(combined_data['时间（h）'], combined_data['园区B弃电量(kW)'], width=bar_width, bottom = -combined_data['园区B弃电量(kW)'], label='Park B Excess (kW)', color='gray', alpha=alpha)
    
    axes[1].set_title('Park B', fontsize=title_fontsize)
    axes[1].set_ylabel('Power (kW)', fontsize=label_fontsize)
    # axes[1].legend(loc='upper right', fontsize=legend_fontsize)
    axes[1].tick_params(axis='both', which='major', labelsize=tick_labelsize)  # 调整刻度标签字体大小

    # 添加阴影效果
    for bar in bars_charge:
        bar.set_edgecolor('grey')
        bar.set_linewidth(1)
        bar.set_alpha(0.7)
        
    # 创建次坐标轴
    ax1 = axes[1].twinx()
    ax1.plot(combined_data['时间（h）'], combined_data['园区B_SOC(kWh)']/storage_capacity, label='Example Curve', color='green', marker='o')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('SOC', fontsize=label_fontsize)
    ax1.tick_params(axis='y', which='major', labelsize=tick_labelsize)  # 调整刻度标签字体大小    
    
    
    # 绘制园区C的数据
    axes[2].bar(combined_data['时间（h）'], combined_data['园区C光伏出力(kW)'], width=bar_width, label='P_V', color='lightpink', alpha=alpha)
    axes[2].bar(combined_data['时间（h）'], combined_data['园区C风电出力(kW)'], width=bar_width, bottom= combined_data['园区C光伏出力(kW)'], label='P_W', color='skyblue', alpha=alpha)
    bars_charge = axes[2].bar(combined_data['时间（h）'], combined_data['园区C_充放电(kWh)'], width=bar_width, bottom= combined_data['园区C负荷(kW)'], label='P_C', color='purple', alpha=1)
    axes[2].bar(combined_data['时间（h）'], combined_data['园区C负荷(kW)'], width=bar_width, label='P_L', edgecolor='black', facecolor='none', linewidth=2, linestyle = '--')
   
    axes[2].bar(combined_data['时间（h）'], combined_data['园区C购电量(kW)'], width=bar_width, bottom= combined_data['园区C光伏出力(kW)']+combined_data['园区C风电出力(kW)'], label='P_G', color='orange', alpha=alpha)
    axes[2].bar(combined_data['时间（h）'], combined_data['园区C弃电量(kW)'], width=bar_width, bottom = -combined_data['园区C弃电量(kW)'], label='P_X', color='gray', alpha=alpha)
    
    axes[2].set_title('Park C', fontsize=title_fontsize)
    axes[2].set_xlabel('Time (h)', fontsize=label_fontsize)
    axes[2].set_ylabel('Power (kW)', fontsize=label_fontsize)
    # axes[2].legend(loc='upper right', fontsize=legend_fontsize)
    axes[2].tick_params(axis='both', which='major', labelsize=tick_labelsize)  # 调整刻度标签字体大小

    axes[2].legend(loc='upper center', fontsize=legend_fontsize, bbox_to_anchor=(0.5, 3.82), ncol=6)

    # 添加阴影效果
    for bar in bars_charge:
        bar.set_edgecolor('grey')
        bar.set_linewidth(1)
        bar.set_alpha(0.7)
    
    # 创建次坐标轴
    ax2 = axes[2].twinx()
    ax2.plot(combined_data['时间（h）'], combined_data['园区C_SOC(kWh)']/storage_capacity, label='SOC', color='green', marker='o')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('SOC', fontsize=label_fontsize)
    ax2.tick_params(axis='y', which='major', labelsize=tick_labelsize)  # 调整刻度标签字体大小    



    # 调整布局
    plt.tight_layout()
    plt.show()
    
    
    
def picture13(combined_data):
    """
    
    """

    # 创建子图
    fig, axes = plt.subplots(1, 1, figsize=(15, 10), sharex=True)
    
    # 定义柱状图的宽度和透明度
    bar_width = 0.7
    alpha = 0.5
    
    # 字体设置
    title_fontsize = 26
    label_fontsize = 24
    legend_fontsize = 25
    tick_labelsize = 24
    
    
    # 绘制园区C的数据
    axes.bar(combined_data['时间（h）'], combined_data['联合园区光伏出力(kW)'], width=bar_width, label='P_V', color='lightpink', alpha=alpha)
    axes.bar(combined_data['时间（h）'], combined_data['联合园区风电出力(kW)'], width=bar_width, bottom= combined_data['联合园区光伏出力(kW)'], label='P_W', color='skyblue', alpha=alpha)
    axes.bar(combined_data['时间（h）'], combined_data['联合园区负荷(kW)'], width=bar_width, label='P_L', edgecolor='black', facecolor='none', linewidth=2, linestyle = '--')
    axes.bar(combined_data['时间（h）'], combined_data['联合园区购电量(kW)'], width=bar_width, bottom= combined_data['联合园区光伏出力(kW)']+combined_data['联合园区风电出力(kW)'], label='P_G', color='orange', alpha=alpha)
    axes.bar(combined_data['时间（h）'], combined_data['联合园区弃电量(kW)'], width=bar_width, bottom = -combined_data['联合园区弃电量(kW)'], label='P_X', color='gray', alpha=alpha)
    
    axes.set_title('Park Joint', fontsize=title_fontsize)
    axes.set_xlabel('Time (h)', fontsize=label_fontsize)
    axes.set_ylabel('Power (kW)', fontsize=label_fontsize)
    axes.legend(loc='upper right', fontsize=legend_fontsize)
    axes.tick_params(axis='both', which='major', labelsize=tick_labelsize)  # 调整刻度标签字体大小

    
    # 调整布局
    plt.tight_layout()
    plt.show()
        

# # 分析未配置储能的经济性
# park_A_result_without_storage = analyze_without_storage('园区A负荷(kW)', '园区A光伏出力(kW)', 1, 0.4)
# park_B_result_without_storage = analyze_without_storage('园区B负荷(kW)', '园区B风电出力(kW)', 1, 0.5)
# park_C_result_without_storage = analyze_without_storage('园区C负荷(kW)', ['园区C光伏出力(kW)', '园区C风电出力(kW)'], 1, [0.4, 0.5])

# park_A_result_without_storage[0].sum()
# park_A_result_without_storage[1].sum()

# combined_data['园区A购电量(kW)'] = park_A_result_without_storage[0]
# combined_data['园区B购电量(kW)'] = park_B_result_without_storage[0]
# combined_data['园区C购电量(kW)'] = park_C_result_without_storage[0]

# combined_data['园区A弃电量(kW)'] = park_A_result_without_storage[1]
# combined_data['园区B弃电量(kW)'] = park_B_result_without_storage[1]
# combined_data['园区C弃电量(kW)'] = park_C_result_without_storage[1]

# picture11(combined_data)



# 分析各园区配置储能后的经济性
park_A_result_with_storage = analyze_with_storage('园区A', '园区A负荷(kW)', '园区A光伏出力(kW)', 1, 0.4)
park_B_result_with_storage = analyze_with_storage('园区B', '园区B负荷(kW)', '园区B风电出力(kW)', 1, 0.5)
park_C_result_with_storage = analyze_with_storage('园区C', '园区C负荷(kW)', ['园区C光伏出力(kW)', '园区C风电出力(kW)'], 1, [0.4, 0.5])

combined_data['园区A购电量(kW)'] = park_A_result_with_storage[0]
combined_data['园区B购电量(kW)'] = park_B_result_with_storage[0]
combined_data['园区C购电量(kW)'] = park_C_result_with_storage[0]

combined_data['园区A弃电量(kW)'] = park_A_result_with_storage[1]
combined_data['园区B弃电量(kW)'] = park_B_result_with_storage[1]
combined_data['园区C弃电量(kW)'] = park_C_result_with_storage[1]


combined_data['园区A_SOC(kWh)'] = park_A_result_with_storage[4]
combined_data['园区B_SOC(kWh)'] = park_B_result_with_storage[4]
combined_data['园区C_SOC(kWh)'] = park_C_result_with_storage[4]

combined_data['园区A_净负荷(kWh)'] = park_A_result_with_storage[5]
combined_data['园区B_净负荷(kWh)'] = park_B_result_with_storage[5]
combined_data['园区C_净负荷(kWh)'] = park_C_result_with_storage[5]

combined_data['园区A_放电(kWh)'] = park_A_result_with_storage[6]
combined_data['园区B_放电(kWh)'] = park_B_result_with_storage[6]
combined_data['园区C_放电(kWh)'] = park_C_result_with_storage[6]

combined_data['园区A_充电(kWh)'] = park_A_result_with_storage[7]
combined_data['园区B_充电(kWh)'] = park_B_result_with_storage[7]
combined_data['园区C_充电(kWh)'] = park_C_result_with_storage[7]

combined_data['园区A_充放电(kWh)'] = combined_data[['园区A_放电(kWh)', '园区A_充电(kWh)']].sum(axis=1)
combined_data['园区B_充放电(kWh)'] = combined_data[['园区B_放电(kWh)', '园区B_充电(kWh)']].sum(axis=1)
combined_data['园区C_充放电(kWh)'] = combined_data[['园区C_放电(kWh)', '园区C_充电(kWh)']].sum(axis=1)

picture12(combined_data)



# # 打印各园区的分析结果
# print("Park A")
# print(park_A_result_without_storage[2])
# print(park_A_result_with_storage[2])

# print("Park B")
# print(park_B_result_without_storage[2])
# print(park_B_result_with_storage[2])

# print("Park C")
# print(park_C_result_without_storage[2])
# print(park_C_result_with_storage[2])

# print("Park A")
# print(park_A_result_without_storage[3])
# print(park_A_result_with_storage[3])

# print("Park B")
# print(park_B_result_without_storage[3])
# print(park_B_result_with_storage[3])

# print("Park C")
# print(park_C_result_without_storage[3])
# print(park_C_result_with_storage[3])

# # 解释各园区的经济性变化原因
# explain_results("Park A", park_A_result_without_storage, park_A_result_with_storage)
# explain_results("Park B", park_B_result_without_storage, park_B_result_with_storage)
# explain_results("Park C", park_C_result_without_storage, park_C_result_with_storage)



# # 联合园区
# joint_result_without_storage = analyze_without_storage(
#     '联合园区负荷(kW)', 
#     ['联合园区光伏出力(kW)', '联合园区风电出力(kW)'], 
#     1, 
#     [0.4, 0.5]
# )

# combined_data['联合园区购电量(kW)'] = joint_result_without_storage[0]
# combined_data['联合园区弃电量(kW)'] = joint_result_without_storage[1]

# joint_purchase = joint_result_without_storage[0].sum()
# joint_excess_generation = joint_result_without_storage[1].sum()
# joint_total_cost = joint_result_without_storage[2]
# joint_avg_cost = joint_result_without_storage[3]
# total_joint = joint_result_without_storage[4]

# picture13(combined_data)
