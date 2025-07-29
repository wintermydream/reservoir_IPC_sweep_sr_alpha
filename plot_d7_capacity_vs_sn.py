"""
绘制不同alpha和sr参数下的容量与噪声水平的关系图。

该脚本从debug_capacity_data/目录加载预先计算的容量数据，
并创建三种不同的图表布局：
1. 2x3的子图布局，分别展示alpha=0.8和1.0，以及sr=0.6、1.0和1.4的组合
2. 2x4的子图布局，每个子图显示特定的degree和alpha组合，并在每个子图中绘制sr=0.6、1.0和1.4三条曲线
3. 3x4的子图布局，每个子图显示特定的sr值和degree组合，并在每个子图中绘制alpha=0.6、0.8和1.0三条曲线
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_capacity_data(out_dir: Path, num_nodes: int, sr: float, alpha: float, omega: float = 1.0) -> tuple:
    """加载指定参数的容量数据。"""
    out_npz = out_dir / f"d1_d3_d5_d7_capacity_vs_sn_N{num_nodes}_sr{sr:.3f}_al{alpha:.3f}_om{omega:.3f}.npz"
    
    if not out_npz.exists():
        print(f"[Error] 找不到数据文件: {out_npz}")
        return None, None, None, None, None
    
    data = np.load(out_npz)
    sn_values = data['sn_values']
    d1_scores = data['d1_scores']
    d3_scores = data['d3_scores']
    d5_scores = data['d5_scores']
    d7_scores = data['d7_scores']
    
    return sn_values, d1_scores, d3_scores, d5_scores, d7_scores

def plot_capacity_vs_sn(ax, sn_values, d1_scores, d3_scores, d5_scores, d7_scores, title):
    """在指定的轴上绘制容量与噪声水平的关系图。"""
    ax.plot(sn_values, d1_scores, 'o-', label='degree 1', markersize=4)
    ax.plot(sn_values, d3_scores, 's--', label='degree 3', markersize=4)
    ax.plot(sn_values, d5_scores, 'd:', label='degree 5', markersize=4)
    ax.plot(sn_values, d7_scores, '^-', label='degree 7', markersize=4)
    
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Capacity')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

def plot_sr_comparison(ax, sn_values_list, scores_list, srs, degree_idx, alpha, title):
    """在指定的轴上绘制不同sr值下的容量与噪声水平的关系图。"""
    markers = ['o-', 's--', 'd:']
    for i, sr in enumerate(srs):
        if sn_values_list[i] is not None:
            # 根据degree_idx选择对应的分数
            if degree_idx == 0:
                scores = scores_list[i][0]  # d1_scores
            elif degree_idx == 1:
                scores = scores_list[i][1]  # d3_scores
            elif degree_idx == 2:
                scores = scores_list[i][2]  # d5_scores
            else:
                scores = scores_list[i][3]  # d7_scores
                
            ax.plot(sn_values_list[i], scores, markers[i], label=f'sr={sr:.1f}', markersize=4)
    
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Capacity')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

def plot_alpha_comparison(ax, sn_values_list, scores_list, alphas, degree_idx, sr, title):
    """在指定的轴上绘制不同alpha值下的容量与噪声水平的关系图。"""
    markers = ['o-', 's--', 'd:']
    for i, alpha in enumerate(alphas):
        if sn_values_list[i] is not None:
            # 根据degree_idx选择对应的分数
            if degree_idx == 0:
                scores = scores_list[i][0]  # d1_scores
            elif degree_idx == 1:
                scores = scores_list[i][1]  # d3_scores
            elif degree_idx == 2:
                scores = scores_list[i][2]  # d5_scores
            else:
                scores = scores_list[i][3]  # d7_scores
                
            ax.plot(sn_values_list[i], scores, markers[i], label=f'α={alpha:.1f}', markersize=4)
    
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Capacity')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

def main() -> None:
    # 参数设置
    num_nodes = 100
    omega = 1.0
    alphas = [0.8, 1.0]
    all_alphas = [0.6, 0.8, 1.0]  # 包含alpha=0.6
    srs = [0.6, 1.0, 1.4]
    out_dir = Path('debug_capacity_data')
    
    # 第一种布局：2x3的子图，每个子图显示特定的alpha和sr组合下的所有degree
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8))
    fig1.suptitle('Capacity vs Noise Level for Different Parameters', fontsize=16)
    
    # 遍历所有参数组合
    for i, alpha in enumerate(alphas):
        for j, sr in enumerate(srs):
            # 加载数据
            sn_values, d1_scores, d3_scores, d5_scores, d7_scores = load_capacity_data(
                out_dir, num_nodes, sr, alpha, omega
            )
            
            if sn_values is None:
                axes1[i, j].text(0.5, 0.5, f"No data for sr={sr}, alpha={alpha}", 
                               ha='center', va='center')
                continue
            
            # 绘制子图
            title = f"sr={sr:.1f}, α={alpha:.1f}"
            plot_capacity_vs_sn(
                axes1[i, j], sn_values, d1_scores, d3_scores, d5_scores, d7_scores, title
            )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为总标题留出空间
    
    # 第二种布局：2x4的子图，每个子图显示特定的degree和alpha组合下的所有sr
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
    fig2.suptitle('SR Comparison for Different Degrees and Alpha Values', fontsize=16)
    
    degrees = [1, 3, 5, 7]
    degree_names = ['Degree 1', 'Degree 3', 'Degree 5', 'Degree 7']
    
    # 预加载所有数据
    data_cache = {}
    for alpha in alphas:
        for sr in srs:
            key = (alpha, sr)
            data_cache[key] = load_capacity_data(out_dir, num_nodes, sr, alpha, omega)
    
    # 遍历所有degree和alpha组合
    for i, alpha in enumerate(alphas):
        for j, degree_idx in enumerate(range(4)):
            # 为每个sr收集数据
            sn_values_list = []
            scores_list = []
            
            for sr in srs:
                data = data_cache[(alpha, sr)]
                sn_values_list.append(data[0])  # sn_values
                # 将所有分数打包在一起
                if data[0] is not None:
                    scores_list.append([data[1], data[2], data[3], data[4]])  # [d1_scores, d3_scores, d5_scores, d7_scores]
                else:
                    scores_list.append(None)
            
            # 绘制子图
            title = f"{degree_names[degree_idx]}, α={alpha:.1f}"
            plot_sr_comparison(
                axes2[i, j], sn_values_list, scores_list, srs, degree_idx, alpha, title
            )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为总标题留出空间
    
    # 第三种布局：3x4的子图，每个子图显示特定的sr和degree组合下的所有alpha
    fig3, axes3 = plt.subplots(3, 4, figsize=(20, 15))
    fig3.suptitle('Alpha Comparison for Different SR Values and Degrees', fontsize=16)
    
    # 预加载所有数据（包括alpha=0.6）
    all_data_cache = {}
    for alpha in all_alphas:
        for sr in srs:
            key = (alpha, sr)
            all_data_cache[key] = load_capacity_data(out_dir, num_nodes, sr, alpha, omega)
    
    # 遍历所有sr和degree组合
    for i, sr in enumerate(srs):
        for j, degree_idx in enumerate(range(4)):
            # 为每个alpha收集数据
            sn_values_list = []
            scores_list = []
            
            for alpha in all_alphas:
                data = all_data_cache[(alpha, sr)]
                sn_values_list.append(data[0])  # sn_values
                # 将所有分数打包在一起
                if data[0] is not None:
                    scores_list.append([data[1], data[2], data[3], data[4]])  # [d1_scores, d3_scores, d5_scores, d7_scores]
                else:
                    scores_list.append(None)
            
            # 绘制子图
            title = f"{degree_names[degree_idx]}, sr={sr:.1f}"
            plot_alpha_comparison(
                axes3[i, j], sn_values_list, scores_list, all_alphas, degree_idx, sr, title
            )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为总标题留出空间
    
    
    # ================== 新增：bar plot 1（alpha=1.0, sr=0.6/1.0/1.4） =====================
    DEGREE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # d1-d7
    HATCHES = ["////", "\\\\", ".."]
    # 1. alpha=1.0, sr=0.6/1.0/1.4
    alpha_bar = 1.0
    srs_bar = [0.6, 1.0, 1.4]
    bar_data = []
    sn_grid = None
    for sr in srs_bar:
        sn_values, d1, d3, d5, d7 = load_capacity_data(out_dir, num_nodes, sr, alpha_bar, omega)
        if sn_values is not None:
            if sn_grid is None:
                sn_grid = sn_values
            bar_data.append((sr, d1, d3, d5, d7))
    if bar_data and sn_grid is not None:
        fig_bar1, ax_bar1 = plt.subplots(figsize=(15, 6))
        n_group = len(bar_data)
        group_gap = 0.05
        bar_width = (0.8 - group_gap) / n_group
        x = np.arange(len(sn_grid))
        legends_handles = []
        for j, (sr, d1, d3, d5, d7) in enumerate(bar_data):
            bar_positions = x - 0.4 + j * bar_width + (j * group_gap / n_group)
            bottom = np.zeros_like(sn_grid, dtype=float)
            for k, (scores, color) in enumerate(zip([d1, d3, d5, d7], DEGREE_COLORS)):
                bars = ax_bar1.bar(
                    bar_positions,
                    scores,
                    bar_width,
                    bottom=bottom,
                    color=color,
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.8,
                    hatch=HATCHES[j % len(HATCHES)],
                )
                if k == 0:
                    legends_handles.append(bars)
                bottom += scores
        ax_bar1.set_xlabel("Noise level (sn)")
        ax_bar1.set_ylabel("Capacity")
        ax_bar1.set_title("Degree 1/3/5/7 Capacity vs Noise Level (alpha=1.0)")
        ax_bar1.set_xticks(x)
        ax_bar1.set_xticklabels([f"{sn:.2f}" for sn in sn_grid])
        ax_bar1.grid(True, linestyle="--", alpha=0.6)
        ax_bar1.legend(legends_handles, [f"sr={sr}" for sr, *_ in bar_data], title="Spectral radius", fontsize=9)
        fig_bar1.tight_layout()
        

    # ================== 新增：bar plot 2（sr=1.0, alpha=0.6/0.8/1.0） =====================
    sr_bar = 1.0
    alphas_bar = [0.6, 0.8, 1.0]
    bar_data2 = []
    sn_grid2 = None
    for alpha in alphas_bar:
        sn_values, d1, d3, d5, d7 = load_capacity_data(out_dir, num_nodes, sr_bar, alpha, omega)
        if sn_values is not None:
            if sn_grid2 is None:
                sn_grid2 = sn_values
            bar_data2.append((alpha, d1, d3, d5, d7))
    if bar_data2 and sn_grid2 is not None:
        fig_bar2, ax_bar2 = plt.subplots(figsize=(15, 6))
        n_group = len(bar_data2)
        group_gap = 0.05
        bar_width = (0.8 - group_gap) / n_group
        x = np.arange(len(sn_grid2))
        legends_handles2 = []
        for j, (alpha, d1, d3, d5, d7) in enumerate(bar_data2):
            bar_positions = x - 0.4 + j * bar_width + (j * group_gap / n_group)
            bottom = np.zeros_like(sn_grid2, dtype=float)
            for k, (scores, color) in enumerate(zip([d1, d3, d5, d7], DEGREE_COLORS)):
                bars = ax_bar2.bar(
                    bar_positions,
                    scores,
                    bar_width,
                    bottom=bottom,
                    color=color,
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.8,
                    hatch=HATCHES[j % len(HATCHES)],
                )
                if k == 0:
                    legends_handles2.append(bars)
                bottom += scores
        ax_bar2.set_xlabel("Noise level (sn)")
        ax_bar2.set_ylabel("Capacity")
        ax_bar2.set_title("Degree 1/3/5/7 Capacity vs Noise Level (sr=1.0)")
        ax_bar2.set_xticks(x)
        ax_bar2.set_xticklabels([f"{sn:.2f}" for sn in sn_grid2])
        ax_bar2.grid(True, linestyle="--", alpha=0.6)
        ax_bar2.legend(legends_handles2, [f"alpha={alpha}" for alpha, *_ in bar_data2], title="Input scaling", fontsize=9)
        fig_bar2.tight_layout()

    # ================== 新增：bar plot 2（sr=0.6, alpha=0.6/0.8/1.0） =====================
    sr_bar = 0.6
    alphas_bar = [0.6, 0.8, 1.0]
    bar_data3 = []
    sn_grid3 = None
    for alpha in alphas_bar:
        sn_values, d1, d3, d5, d7 = load_capacity_data(out_dir, num_nodes, sr_bar, alpha, omega)
        if sn_values is not None:
            if sn_grid3 is None:
                sn_grid3 = sn_values
            bar_data3.append((alpha, d1, d3, d5, d7))
    if bar_data3 and sn_grid3 is not None:
        fig_bar3, ax_bar3 = plt.subplots(figsize=(15, 6))
        n_group = len(bar_data3)
        group_gap = 0.05
        bar_width = (0.8 - group_gap) / n_group
        x = np.arange(len(sn_grid3))
        legends_handles3 = []
        for j, (alpha, d1, d3, d5, d7) in enumerate(bar_data3):
            bar_positions = x - 0.4 + j * bar_width + (j * group_gap / n_group)
            bottom = np.zeros_like(sn_grid3, dtype=float)
            for k, (scores, color) in enumerate(zip([d1, d3, d5, d7], DEGREE_COLORS)):
                bars = ax_bar3.bar(
                    bar_positions,
                    scores,
                    bar_width,
                    bottom=bottom,
                    color=color,
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.8,
                    hatch=HATCHES[j % len(HATCHES)],
                )
                if k == 0:
                    legends_handles3.append(bars)
                bottom += scores
        ax_bar3.set_xlabel("Noise level (sn)")
        ax_bar3.set_ylabel("Capacity")
        ax_bar3.set_title("Degree 1/3/5/7 Capacity vs Noise Level (sr=0.6)")
        ax_bar3.set_xticks(x)
        ax_bar3.set_xticklabels([f"{sn:.2f}" for sn in sn_grid3])
        ax_bar3.grid(True, linestyle="--", alpha=0.6)
        ax_bar3.legend(legends_handles3, [f"alpha={alpha}" for alpha, *_ in bar_data3], title="Input scaling", fontsize=9)
        fig_bar3.tight_layout()

    # ================== 新增：bar plot 2（sr=1.4, alpha=0.6/0.8/1.0） =====================
    sr_bar = 1.4
    alphas_bar = [0.6, 0.8, 1.0]
    bar_data4 = []
    sn_grid4 = None
    for alpha in alphas_bar:
        sn_values, d1, d3, d5, d7 = load_capacity_data(out_dir, num_nodes, sr_bar, alpha, omega)
        if sn_values is not None:
            if sn_grid4 is None:
                sn_grid4 = sn_values
            bar_data4.append((alpha, d1, d3, d5, d7))
    if bar_data4 and sn_grid4 is not None:
        fig_bar4, ax_bar4 = plt.subplots(figsize=(15, 6))
        n_group = len(bar_data4)
        group_gap = 0.05
        bar_width = (0.8 - group_gap) / n_group
        x = np.arange(len(sn_grid4))
        legends_handles4 = []
        for j, (alpha, d1, d3, d5, d7) in enumerate(bar_data4):
            bar_positions = x - 0.4 + j * bar_width + (j * group_gap / n_group)
            bottom = np.zeros_like(sn_grid4, dtype=float)
            for k, (scores, color) in enumerate(zip([d1, d3, d5, d7], DEGREE_COLORS)):
                bars = ax_bar4.bar(
                    bar_positions,
                    scores,
                    bar_width,
                    bottom=bottom,
                    color=color,
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.8,
                    hatch=HATCHES[j % len(HATCHES)],
                )
                if k == 0:
                    legends_handles4.append(bars)
                bottom += scores
        ax_bar4.set_xlabel("Noise level (sn)")
        ax_bar4.set_ylabel("Capacity")
        ax_bar4.set_title("Degree 1/3/5/7 Capacity vs Noise Level (sr=1.4)")
        ax_bar4.set_xticks(x)
        ax_bar4.set_xticklabels([f"{sn:.2f}" for sn in sn_grid4])
        ax_bar4.grid(True, linestyle="--", alpha=0.6)
        ax_bar4.legend(legends_handles4, [f"alpha={alpha}" for alpha, *_ in bar_data4], title="Input scaling", fontsize=9)
        fig_bar4.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()