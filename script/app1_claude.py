import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import io
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from matplotlib import rcParams
import scipy 
import os
import glob

st.set_page_config(page_title="Episode文件曲线编辑", layout="wide")
st.title("✏️ Episode文件直接手绘修改")

# 初始化session state
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'modified_data' not in st.session_state:
    st.session_state.modified_data = None
if 'plot_bounds' not in st.session_state:
    st.session_state.plot_bounds = None
if 'edit_column' not in st.session_state:
    st.session_state.edit_column = 0
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'current_episode' not in st.session_state:
    st.session_state.current_episode = None

def validate_data(data, data_name):
    """验证数据的有效性"""
    if data is None or len(data) == 0:
        return False, f"{data_name}为空"
    if len(data) < 2:
        return False, f"{data_name}数据点少于2个"
    if not np.isfinite(data).all():
        return False, f"{data_name}包含无效值(NaN或无穷大)"
    return True, f"{data_name}验证通过: {len(data)}个数据点"

def detect_change_rate_events(data, sample_rate, window_size_sec, threshold):
    """检测轨迹在给定时间窗口内发生剧烈变化的起始时刻"""
    data = np.asarray(data)
    window_size = int(window_size_sec * sample_rate)
    if window_size < 1:
        raise ValueError("window_size_sec * sample_rate must be ≥ 1")

    events_idx = []
    rates = []
    for i in range(len(data) - window_size):
        diff = data[i + window_size] - data[i]
        rate = abs(diff) 
        if rate > threshold:
            events_idx.append(i)
            rates.append(rate)

    events_sec = [idx / sample_rate for idx in events_idx]
    return events_sec, rates

def get_plot_bounds(fig, ax):
    """获取matplotlib绘图区域的实际像素边界"""
    dpi = fig.dpi
    fig_width, fig_height = fig.get_size_inches()
    bbox = ax.get_position()
    
    left = bbox.x0 * fig_width * dpi
    bottom = (1 - bbox.y1) * fig_height * dpi
    width = bbox.width * fig_width * dpi
    height = bbox.height * fig_height * dpi
    
    return {
        'left': left,
        'right': left + width,
        'top': bottom,
        'bottom': bottom + height,
        'width': width,
        'height': height
    }

def smooth_path_points(points, smooth_factor=1.0):
    """对路径点进行平滑处理"""
    if len(points) < 3:
        return points
    
    xs, ys = zip(*points)
    xs = np.array(xs)
    ys = np.array(ys)
    
    if smooth_factor > 0:
        ys_smooth = gaussian_filter1d(ys, sigma=smooth_factor)
        return list(zip(xs, ys_smooth))
    
    return points

def interpolate_modification(original_data, path_points, plot_bounds, data_bounds):
    """插值修改函数 - 调试版本"""
    if not path_points:
        return original_data.copy()
    
    smoothed_points = smooth_path_points(path_points, smooth_factor=1.0)
    xs, ys = zip(*smoothed_points)
    xs = np.array(xs)
    ys = np.array(ys)
    
    print(f"\n=== 插值调试信息 ===")
    print(f"绘制点数: {len(path_points)}")
    print(f"画布坐标范围: x=[{xs.min():.1f}, {xs.max():.1f}], y=[{ys.min():.1f}, {ys.max():.1f}]")
    
    # 坐标映射
    x_data_min, x_data_max = data_bounds['x_min'], data_bounds['x_max']
    t_mapped = np.interp(xs, [plot_bounds['left'], plot_bounds['right']], [x_data_min, x_data_max])
    
    y_data_min, y_data_max = data_bounds['y_min'], data_bounds['y_max']
    a_mapped = np.interp(ys, [plot_bounds['top'], plot_bounds['bottom']], [y_data_max, y_data_min])
    
    print(f"映射后数据坐标: t=[{t_mapped.min():.1f}, {t_mapped.max():.1f}], a=[{a_mapped.min():.1f}, {a_mapped.max():.1f}]")
    
    # 检查第一个点的映射
    if len(xs) > 0:
        print(f"第一个点映射: 画布({xs[0]:.1f}, {ys[0]:.1f}) -> 数据({t_mapped[0]:.1f}, {a_mapped[0]:.1f})")
        print(f"预期时间索引: {int(round(t_mapped[0]))}")
    
    modified = original_data.copy()
    if not isinstance(modified, pd.Series):
        modified = pd.Series(modified, dtype=float)
    
    modified_points = []
    
    if len(t_mapped) > 0:
        try:
            if len(t_mapped) > 1:
                sorted_indices = np.argsort(t_mapped)
                t_sorted = t_mapped[sorted_indices]
                a_sorted = a_mapped[sorted_indices]
                
                unique_mask = np.concatenate(([True], np.diff(t_sorted) > 1e-6))
                t_unique = t_sorted[unique_mask]
                a_unique = a_sorted[unique_mask]
                
                if len(t_unique) > 1:
                    interp_func = interp1d(t_unique, a_unique, kind='linear', 
                                         bounds_error=False, fill_value='extrapolate')
                    
                    t_start = max(0, int(np.floor(t_mapped.min())))
                    t_end = min(len(original_data) - 1, int(np.ceil(t_mapped.max())))
                    
                    print(f"修改范围: {t_start} 到 {t_end}")
                    
                    for t_second in range(t_start, t_end + 1):
                        if 0 <= t_second < len(modified):
                            old_value = modified.iloc[t_second]
                            new_angle = float(interp_func(t_second))
                            modified.iloc[t_second] = new_angle
                            modified_points.append((t_second, old_value, new_angle))
                            
                            # 打印前几个修改点
                            if len(modified_points) <= 3:
                                print(f"修改点 {t_second}: {old_value:.2f} -> {new_angle:.2f}")
                else:
                    t_second = int(round(t_unique[0]))
                    if 0 <= t_second < len(modified):
                        old_value = modified.iloc[t_second]
                        new_angle = float(a_unique[0])
                        modified.iloc[t_second] = new_angle
                        modified_points.append((t_second, old_value, new_angle))
                        print(f"单点修改 {t_second}: {old_value:.2f} -> {new_angle:.2f}")
            else:
                t_second = int(round(t_mapped[0]))
                if 0 <= t_second < len(modified):
                    old_value = modified.iloc[t_second]
                    new_angle = float(a_mapped[0])
                    modified.iloc[t_second] = new_angle
                    modified_points.append((t_second, old_value, new_angle))
                    print(f"单点修改 {t_second}: {old_value:.2f} -> {new_angle:.2f}")
                    
        except Exception as e:
            print(f"插值处理出错: {str(e)}")
            for t_val, a_val in zip(t_mapped, a_mapped):
                t_second = int(round(t_val))
                if 0 <= t_second < len(modified):
                    old_value = modified.iloc[t_second]
                    new_angle = float(a_val)
                    modified.iloc[t_second] = new_angle
                    modified_points.append((t_second, old_value, new_angle))
    
    print(f"总共修改了 {len(modified_points)} 个点")
    print("==================\n")
    
    return modified
def create_background_image(time_data, angle_data, canvas_width=1000, canvas_height=600, change_threshold=30.0, window_size=3):
    """创建背景图像 - 调试版本，打印坐标信息，扩大纵轴范围"""
    
    # 固定参数
    dpi = 100
    fig_width = canvas_width / dpi
    fig_height = canvas_height / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    # 设置更精确的绘图区域
    left_margin = 80 / canvas_width      # 80像素左边距
    right_margin = 50 / canvas_width     # 50像素右边距  
    bottom_margin = 80 / canvas_height   # 80像素下边距
    top_margin = 50 / canvas_height      # 50像素上边距
    
    plot_left = left_margin
    plot_right = 1 - right_margin
    plot_bottom = bottom_margin
    plot_top = 1 - top_margin
    
    ax.set_position([plot_left, plot_bottom, 
                     plot_right - plot_left, 
                     plot_top - plot_bottom])
    
    # 绘制曲线
    ax.plot(time_data, angle_data, color='blue', linewidth=2, label='原始曲线', alpha=0.8)
    
    # 检测变化率事件
    events, rates = detect_change_rate_events(
        angle_data, sample_rate=1.0, window_size_sec=window_size, threshold=change_threshold
    )
    
    if events:
        for event_time, rate in zip(events, rates):
            ax.axvline(x=event_time, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
            y_pos = angle_data.iloc[int(event_time)] if int(event_time) < len(angle_data) else angle_data.iloc[-1]
            ax.annotate(f'{rate:.1f}°/s', xy=(event_time, y_pos), xytext=(5, 10), 
                       textcoords='offset points', fontsize=8, color='orange', alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        ax.plot([], [], color='orange', linestyle='--', alpha=0.7, 
                label=f'变化率>{change_threshold}°/{window_size}s')
    
    ax.set_xlabel("时间 (s)", fontsize=12)
    ax.set_ylabel("数值", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("在曲线上绘制修改（红色）", fontsize=14)
    
    # === 修改部分：扩大纵轴范围 ===
    # 计算原始数据的范围
    y_min_original = angle_data.min()
    y_max_original = angle_data.max()
    y_center = (y_max_original + y_min_original) / 2
    y_range_original = y_max_original - y_min_original
    
    # 扩大系数（可以根据需要调整）
    expansion_factor = 4.0  # 扩大2倍范围
    
    # 计算扩大后的范围，保持对称
    expanded_range = y_range_original * expansion_factor
    y_min_expanded = y_center - expanded_range / 2
    y_max_expanded = y_center + expanded_range / 2
    
    # 如果原始范围太小，设置最小扩展范围
    min_range = 50.0  # 最小范围，可根据数据特点调整
    if expanded_range < min_range:
        y_min_expanded = y_center - min_range / 2
        y_max_expanded = y_center + min_range / 2
    
    # 强制设置坐标轴范围
    ax.set_xlim(time_data.min(), time_data.max())
    ax.set_ylim(y_min_expanded, y_max_expanded)  # 使用扩大后的范围
    
    # 计算精确的像素边界
    plot_bounds = {
        'left': plot_left * canvas_width,
        'right': plot_right * canvas_width,
        'top': (1 - plot_top) * canvas_height,     # Y轴翻转
        'bottom': (1 - plot_bottom) * canvas_height,
        'width': (plot_right - plot_left) * canvas_width,
        'height': (plot_top - plot_bottom) * canvas_height
    }
    
    # 使用扩大后的纵轴范围作为数据边界
    data_bounds = {
        'x_min': float(time_data.min()),
        'x_max': float(time_data.max()),
        'y_min': float(y_min_expanded),  # 使用扩大后的最小值
        'y_max': float(y_max_expanded)   # 使用扩大后的最大值
    }
    
    # 打印调试信息
    print("=== 坐标映射调试信息 ===")
    print(f"Canvas尺寸: {canvas_width} x {canvas_height}")
    print(f"绘图区域像素边界: left={plot_bounds['left']:.1f}, right={plot_bounds['right']:.1f}")
    print(f"绘图区域像素边界: top={plot_bounds['top']:.1f}, bottom={plot_bounds['bottom']:.1f}")
    print(f"原始数据Y范围: [{y_min_original:.1f}, {y_max_original:.1f}] (范围: {y_range_original:.1f})")
    print(f"扩大后Y范围: [{y_min_expanded:.1f}, {y_max_expanded:.1f}] (范围: {expanded_range:.1f})")
    print(f"数据范围: x=[{data_bounds['x_min']:.1f}, {data_bounds['x_max']:.1f}]")
    print(f"数据范围: y=[{data_bounds['y_min']:.1f}, {data_bounds['y_max']:.1f}]")
    print("=======================")
    
    # 保存图像
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, facecolor='white', edgecolor='none', pad_inches=0)
    buf.seek(0)
    background = Image.open(buf)
    
    if background.size != (canvas_width, canvas_height):
        background = background.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
    
    plt.close(fig)
    
    return background, plot_bounds, data_bounds, events, rates

def extract_path_points(canvas_result):
    """从画布结果中提取所有路径点"""
    if not canvas_result.json_data or not canvas_result.json_data.get("objects"):
        return []
    
    all_points = []
    
    for obj in canvas_result.json_data["objects"]:
        if obj["type"] == "path" and "path" in obj:
            path_points = []
            
            for segment in obj["path"]:
                if len(segment) >= 3:
                    cmd = segment[0]
                    if cmd in ("M", "L"):
                        x, y = segment[1], segment[2]
                        path_points.append((x, y))
                    elif cmd == "Q":
                        if len(segment) >= 5:
                            x, y = segment[3], segment[4]
                            path_points.append((x, y))
                    elif cmd == "C":
                        if len(segment) >= 7:
                            x, y = segment[5], segment[6]
                            path_points.append((x, y))
            
            if path_points:
                all_points.extend(path_points)
    
    return all_points

def safe_update_dataframe_column(output_df, current_column, modified_data):
    """安全地更新DataFrame列的函数 - 精确修复版本"""
    try:
        # 获取修改数据的值
        if isinstance(modified_data, pd.Series):
            modified_values = modified_data.values
        elif isinstance(modified_data, np.ndarray):
            modified_values = modified_data
        else:
            modified_values = np.array(modified_data, dtype=float)
        
        # 确保长度匹配
        df_length = len(output_df)
        data_length = len(modified_values)
        
        if data_length != df_length:
            return False, f"数据长度不匹配: DataFrame={df_length}, 修改数据={data_length}"
        
        # 检查原始数据格式
        sample_value = output_df.iloc[0, current_column]
        print(f"样本值类型: {type(sample_value)}, 值: {sample_value}")
        
        if isinstance(sample_value, (list, tuple, np.ndarray)):
            # 处理列表/数组格式的数据 - 逐行更新，避免批量赋值问题
            print("检测到复杂数据结构，使用逐行更新")
            
            for i in range(len(output_df)):
                old_val = output_df.iloc[i, current_column]
                new_scalar_value = float(modified_values[i])
                
                if isinstance(old_val, list):
                    new_val = old_val.copy()
                    new_val[5] = new_scalar_value
                    output_df.iat[i, current_column] = new_val  # 使用 iat 而不是 iloc
                elif isinstance(old_val, tuple):
                    new_val = list(old_val)
                    new_val[5] = new_scalar_value
                    output_df.iat[i, current_column] = tuple(new_val)
                elif isinstance(old_val, np.ndarray):
                    new_val = old_val.copy()
                    new_val[5] = new_scalar_value
                    output_df.iat[i, current_column] = new_val
                else:
                    # 如果某些行不是预期格式，直接使用新值
                    output_df.iat[i, current_column] = new_scalar_value
            
        else:
            # 处理标量数据 - 可以安全地批量赋值
            print("检测到标量数据，使用批量更新")
            output_df.iloc[:, current_column] = modified_values.astype(float)
        
        return True, f"数据更新成功，处理了 {len(modified_values)} 个数据点"
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return False, f"数据更新失败: {str(e)}\n详细错误:\n{error_details}"


# ==== 参数输入 ====
st.sidebar.header("📁 数据选择")
dataset_name = st.sidebar.text_input("数据集名称", value="uni_softO_handing")
episode_idx = st.sidebar.number_input("Episode 编号", min_value=0, step=1, value=0)

# 列选择
edit_column = st.sidebar.selectbox("选择编辑列", options=[0, 1], format_func=lambda x: f"列{x}", index=st.session_state.edit_column)
st.session_state.edit_column = edit_column

# ==== 路径处理 ====
base_path = f"../clean_data/{dataset_name}/data/chunk-000"
file_path = os.path.join(base_path, f"episode_{episode_idx:06d}.parquet")
save_dir = f"../poisoning_data/{dataset_name}"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"episode_{episode_idx:06d}.parquet")

# 检查是否需要重新加载数据
need_reload = (st.session_state.current_dataset != dataset_name or 
               st.session_state.current_episode != episode_idx)

# ==== 载入数据 ====
if not os.path.exists(file_path):
    st.error(f"未找到文件：{file_path}")
    st.stop()

if need_reload or st.session_state.original_data is None:
    try:
        with st.spinner(f"正在加载 Episode {episode_idx}..."):
            df = pd.read_parquet(file_path).iloc[:, :2]
            col_names = [f"列{i}" for i in range(df.shape[1])]
            current_column = st.session_state.edit_column
            raw_series = df.iloc[:, current_column]

            angle_data = raw_series.apply(lambda x: x[5]).copy()
            angle_data = pd.Series(angle_data, dtype=float)
            
            time_data = np.arange(len(angle_data))
            
            # 更新session state
            st.session_state.original_data = angle_data
            st.session_state.modified_data = None
            st.session_state.current_dataset = dataset_name
            st.session_state.current_episode = episode_idx
            
            st.success(f"成功加载数据集: {dataset_name}, Episode: {episode_idx}")
            
    except Exception as e:
        st.error(f"载入数据失败: {e}")
        st.stop()

# 如果数据已加载，继续处理
if st.session_state.original_data is not None:
    # 重新获取当前列的数据（以防列选择发生变化）
    try:
        df = pd.read_parquet(file_path).iloc[:, :2]
        col_names = [f"列{i}" for i in range(df.shape[1])]
        current_column = st.session_state.edit_column
        raw_series = df.iloc[:, current_column]
        angle_data = raw_series.apply(lambda x: x[5] if isinstance(x, (list, tuple, np.ndarray)) else x).copy()
        angle_data = pd.Series(angle_data, dtype=float)
        time_data = np.arange(len(angle_data))
        
        # 如果列发生变化，更新原始数据
        if not angle_data.equals(st.session_state.original_data):
            st.session_state.original_data = angle_data
            st.session_state.modified_data = None
            
    except Exception as e:
        st.error(f"重新载入数据失败: {e}")
        st.stop()

    st.subheader(f"🎯 当前编辑: {dataset_name} - Episode {episode_idx} - {col_names[current_column]}")
    
    # 显示文件路径信息
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📂 源文件: {file_path}")
    with col2:
        st.info(f"💾 保存路径: {save_path}")
    
    # 数据验证
    is_valid, message = validate_data(angle_data, f"{col_names[current_column]}数据")
    if not is_valid:
        st.error(f"数据验证失败: {message}")
        st.stop()
    else:
        st.success(message)
        
        # 显示数据信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("数据点数", len(angle_data))
        with col2:
            st.metric("最小值", f"{angle_data.min():.2f}")
        with col3:
            st.metric("最大值", f"{angle_data.max():.2f}")
        with col4:
            st.metric("平均值", f"{angle_data.mean():.2f}")
        
        # 参数设置
        st.subheader("🛠️ 绘制参数")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stroke_width = st.slider("画笔宽度", 1, 10, 3)
        with col2:
            smooth_factor = st.slider("平滑程度", 0.0, 3.0, 1.0, 0.1)
        with col3:
            change_threshold = st.slider("变化率阈值", 1.0, 100.0, 30.0, 5.0)
        with col4:
            window_size = st.slider("检测窗口 (秒)", 1, 300, 3)
        
        # 创建背景图像
        canvas_width, canvas_height = 1000, 600
        background, plot_bounds, data_bounds, events, rates = create_background_image(
            time_data, angle_data, 
            #figsize=(canvas_width/100, canvas_height/100),
            change_threshold=change_threshold,
            window_size=window_size
        )
        
        st.session_state.plot_bounds = plot_bounds
        st.session_state.data_bounds = data_bounds
        
        # 显示变化率检测结果
        if events:
            st.subheader("⚡ 检测到的高变化率时刻")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("检测到事件数", len(events))
            with col2:
                st.metric("最大变化率", f"{max(rates):.1f}/s")
            with col3:
                st.metric("平均变化率", f"{np.mean(rates):.1f}/s")
            
            with st.expander("📋 详细事件列表"):
                for i, (event_time, rate) in enumerate(zip(events, rates)):
                    event_idx = int(event_time)
                    if event_idx < len(angle_data):
                        value_at_event = angle_data.iloc[event_idx]
                        st.write(f"事件 {i+1}: 时间 {event_time}s, 数值 {value_at_event:.2f}, 变化率 {rate:.2f}/{window_size}s")
        else:
            st.info(f"💡 在当前阈值下未检测到高变化率事件")
        
        # 绘制画布
        st.subheader("🖍️ 在原图上绘制修改曲线段")
        st.info("💡 提示：用鼠标在蓝色曲线上绘制红色修改线段。橙色虚线表示检测到的高变化率时刻。")
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.1)",
            stroke_width=stroke_width,
            stroke_color="#FF0000",
            background_image=background,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            key=f"canvas_{dataset_name}_{episode_idx}_{current_column}",
        )
        
        # 处理绘制结果
        if canvas_result.json_data:
            path_points = extract_path_points(canvas_result)
            
            if path_points:
                modified_data = interpolate_modification(
                    st.session_state.original_data,
                    path_points,
                    st.session_state.plot_bounds,
                    st.session_state.data_bounds
                )
                
                st.session_state.modified_data = modified_data
                
                # 显示修改信息
                st.subheader("📊 修改统计")
                diff = modified_data - st.session_state.original_data
                changed_points = np.sum(np.abs(diff) > 0.01)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("修改点数", changed_points)
                with col2:
                    st.metric("最大变化", f"{np.abs(diff).max():.2f}")
                with col3:
                    if changed_points > 0:
                        avg_change = np.abs(diff[np.abs(diff) > 0.01]).mean()
                        st.metric("平均变化", f"{avg_change:.2f}")
                    else:
                        st.metric("平均变化", "0.00")
                
                # 对比图表
                st.subheader("📈 修改效果对比")
                df_result = pd.DataFrame({
                    "时间 (s)": time_data,
                    "原始数值": st.session_state.original_data,
                    "修改后数值": modified_data,
                    "差值": diff
                })
                
                st.line_chart(df_result.set_index("时间 (s)")[["原始数值", "修改后数值"]])
                
                st.subheader("📊 修改差值")
                st.line_chart(df_result.set_index("时间 (s)")["差值"])
                
                # 操作按钮
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 重置修改", type="secondary"):
                        st.session_state.modified_data = None
                        st.rerun()
                
                with col2:
                    if st.button("📋 查看数据详情"):
                        st.dataframe(df_result, use_container_width=True)
                
                with col3:
                    if st.button("💾 保存到指定路径", type="primary"):
                        try:
                            # 重新读取原始文件
                            original_df = pd.read_parquet(file_path)
                            
                            # 创建修改后的DataFrame
                            output_df = original_df.copy()
                            
                            # 使用安全的更新函数
                            success, message = safe_update_dataframe_column(
                                output_df, current_column, modified_data
                            )
                            
                            if success:
                                # 保存文件
                                output_df.to_parquet(save_path, index=False)
                                st.success(f"✅ 文件已保存到: {save_path}")
                            else:
                                st.error(f"❌ {message}")
                                
                        except Exception as e:
                            st.error(f"保存失败: {e}")
                            # 添加详细的错误调试信息
                            st.write("调试信息:")
                            st.write(f"modified_data 类型: {type(modified_data)}")
                            st.write(f"modified_data 长度: {len(modified_data)}")
                            st.write(f"DataFrame 行数: {len(original_df)}")
                            st.write(f"当前列索引: {current_column}")
                
                # 下载按钮
                if st.button("📥 下载修改后数据", type="secondary"):
                    try:
                        # 重新读取原始文件用于下载
                        original_df = pd.read_parquet(file_path)
                        output_df = original_df.copy()
                        
                        # 使用安全的更新函数
                        success, message = safe_update_dataframe_column(
                            output_df, current_column, modified_data
                        )
                        
                        if success:
                            # 创建下载
                            buffer = io.BytesIO()
                            output_df.to_parquet(buffer, index=False)
                            buffer.seek(0)
                            
                            st.download_button(
                                label="📄 下载修改后的Parquet文件",
                                data=buffer,
                                file_name=f"episode_{episode_idx:06d}_modified.parquet",
                                mime="application/octet-stream"
                            )
                        else:
                            st.error(f"❌ 准备下载失败: {message}")
                        
                    except Exception as e:
                        st.error(f"准备下载失败: {e}")
            
            else:
                st.info("👆 请在图上绘制修改路径")