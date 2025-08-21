import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import io
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import os
import gc
import hashlib

# 兼容不同版本的Streamlit
def safe_rerun():
    """安全的重新运行函数，兼容不同版本的Streamlit"""
    try:
        if hasattr(st, 'rerun'):
            st.rerun()
        elif hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
        else:
            # 如果都不存在，使用session state触发重新运行
            if 'rerun_trigger' not in st.session_state:
                st.session_state.rerun_trigger = 0
            st.session_state.rerun_trigger += 1
            st.experimental_rerun()
    except:
        # 最后的fallback，刷新整个页面
        st.write('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)

st.set_page_config(page_title="Episode文件曲线编辑", layout="wide")
st.title("✏️ Episode文件直接手绘修改")

# 改进的session state初始化
def initialize_session_state():
    """初始化session state并添加版本控制"""
    defaults = {
        'original_data': None,
        'modified_data': None,
        'plot_bounds': None,
        'data_bounds': None,
        'edit_column': 0,
        'array_index': 0,
        'current_dataset': None,
        'current_episode': None,
        'current_array_index': None,
        'canvas_key_version': 0,  # 用于强制刷新canvas
        'last_file_hash': None,   # 用于检测文件变化
        'error_count': 0,         # 错误计数
        'max_errors': 3           # 最大错误数
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def clear_session_data():
    """清理session state中的数据"""
    keys_to_clear = ['original_data', 'modified_data', 'plot_bounds', 'data_bounds', 'last_file_hash']
    for key in keys_to_clear:
        if key in st.session_state:
            st.session_state[key] = None
    
    # 强制垃圾回收
    gc.collect()

def get_file_hash(file_path):
    """获取文件的哈希值，用于检测文件变化"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def safe_matplotlib_cleanup():
    """安全地清理matplotlib资源"""
    try:
        plt.close('all')
        plt.clf()
        plt.cla()
    except:
        pass

def validate_data(data, data_name):
    """验证数据的有效性"""
    if data is None or len(data) == 0:
        return False, f"{data_name}为空"
    if len(data) < 2:
        return False, f"{data_name}数据点少于2个"
    if not np.isfinite(data).all():
        return False, f"{data_name}包含无效值"
    return True, f"{data_name}验证通过: {len(data)}个数据点"

def smooth_path_points(points, smooth_factor=1.0):
    """对路径点进行平滑处理"""
    if len(points) < 3:
        return points
    
    try:
        xs, ys = zip(*points)
        xs, ys = np.array(xs), np.array(ys)
        
        if smooth_factor > 0:
            ys_smooth = gaussian_filter1d(ys, sigma=smooth_factor)
            return list(zip(xs, ys_smooth))
        
        return points
    except Exception as e:
        st.error(f"路径平滑处理失败: {e}")
        return points

def interpolate_modification(original_data, path_points, plot_bounds, data_bounds):
    """插值修改函数 - 添加更多错误处理"""
    if not path_points:
        return original_data.copy()
    
    try:
        smoothed_points = smooth_path_points(path_points, smooth_factor=1.0)
        if not smoothed_points:
            return original_data.copy()
            
        xs, ys = zip(*smoothed_points)
        xs, ys = np.array(xs), np.array(ys)
        
        # 检查坐标范围
        if len(xs) == 0 or len(ys) == 0:
            return original_data.copy()
        
        # 坐标映射
        x_data_min, x_data_max = data_bounds['x_min'], data_bounds['x_max']
        y_data_min, y_data_max = data_bounds['y_min'], data_bounds['y_max']
        
        # 添加边界检查
        if plot_bounds['right'] <= plot_bounds['left'] or plot_bounds['bottom'] <= plot_bounds['top']:
            st.error("绘图边界无效")
            return original_data.copy()
            
        t_mapped = np.interp(xs, [plot_bounds['left'], plot_bounds['right']], [x_data_min, x_data_max])
        a_mapped = np.interp(ys, [plot_bounds['top'], plot_bounds['bottom']], [y_data_max, y_data_min])
        
        modified = original_data.copy()
        if not isinstance(modified, pd.Series):
            modified = pd.Series(modified, dtype=float)
        
        if len(t_mapped) > 0:
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
                    
                    for t_second in range(t_start, t_end + 1):
                        if 0 <= t_second < len(modified):
                            try:
                                new_angle = float(interp_func(t_second))
                                if np.isfinite(new_angle):
                                    modified.iloc[t_second] = new_angle
                            except:
                                continue
                else:
                    t_second = int(round(t_unique[0]))
                    if 0 <= t_second < len(modified):
                        new_val = float(a_unique[0])
                        if np.isfinite(new_val):
                            modified.iloc[t_second] = new_val
            else:
                t_second = int(round(t_mapped[0]))
                if 0 <= t_second < len(modified):
                    new_val = float(a_mapped[0])
                    if np.isfinite(new_val):
                        modified.iloc[t_second] = new_val
                        
        return modified
        
    except Exception as e:
        st.error(f"插值处理出错: {str(e)}")
        st.session_state.error_count += 1
        return original_data.copy()

@st.cache_data
def create_background_image(time_data_tuple, angle_data_tuple, canvas_width=1000, canvas_height=600):
    """创建背景图像 - 使用缓存优化"""
    try:
        # 确保清理之前的图形
        safe_matplotlib_cleanup()
        
        time_data = np.array(time_data_tuple)
        angle_data = np.array(angle_data_tuple)
        
        dpi = 100
        fig_width = canvas_width / dpi
        fig_height = canvas_height / dpi
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # 设置绘图区域
        left_margin = 80 / canvas_width
        right_margin = 50 / canvas_width
        bottom_margin = 80 / canvas_height
        top_margin = 50 / canvas_height
        
        plot_left = left_margin
        plot_right = 1 - right_margin
        plot_bottom = bottom_margin
        plot_top = 1 - top_margin
        
        ax.set_position([plot_left, plot_bottom, plot_right - plot_left, plot_top - plot_bottom])
        
        # 绘制曲线
        ax.plot(time_data, angle_data, color='blue', linewidth=2, label='原始曲线', alpha=0.8)
        
        ax.set_xlabel("时间 (s)", fontsize=12)
        ax.set_ylabel("数值", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("在曲线上绘制修改（红色）", fontsize=14)
        
        # 扩大纵轴范围
        y_min_original = angle_data.min()
        y_max_original = angle_data.max()
        y_center = (y_max_original + y_min_original) / 2
        y_range_original = y_max_original - y_min_original
        
        expansion_factor = 4.0
        expanded_range = max(y_range_original * expansion_factor, 50.0)
        y_min_expanded = y_center - expanded_range / 2
        y_max_expanded = y_center + expanded_range / 2
        
        ax.set_xlim(time_data.min(), time_data.max())
        ax.set_ylim(y_min_expanded, y_max_expanded)
        
        # 计算像素边界
        plot_bounds = {
            'left': plot_left * canvas_width,
            'right': plot_right * canvas_width,
            'top': (1 - plot_top) * canvas_height,
            'bottom': (1 - plot_bottom) * canvas_height,
            'width': (plot_right - plot_left) * canvas_width,
            'height': (plot_top - plot_bottom) * canvas_height
        }
        
        data_bounds = {
            'x_min': float(time_data.min()),
            'x_max': float(time_data.max()),
            'y_min': float(y_min_expanded),
            'y_max': float(y_max_expanded)
        }
        
        # 保存图像
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, facecolor='white', edgecolor='none', pad_inches=0)
        buf.seek(0)
        background = Image.open(buf)
        
        if background.size != (canvas_width, canvas_height):
            background = background.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        plt.close(fig)
        
        return background, plot_bounds, data_bounds
        
    except Exception as e:
        st.error(f"背景图像创建失败: {e}")
        # 返回一个空白图像作为fallback
        blank_image = Image.new('RGB', (canvas_width, canvas_height), 'white')
        empty_bounds = {'left': 0, 'right': canvas_width, 'top': 0, 'bottom': canvas_height}
        empty_data_bounds = {'x_min': 0, 'x_max': 1, 'y_min': 0, 'y_max': 1}
        return blank_image, empty_bounds, empty_data_bounds

def extract_path_points(canvas_result):
    """从画布结果中提取所有路径点"""
    if not canvas_result.json_data or not canvas_result.json_data.get("objects"):
        return []
    
    try:
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
                        elif cmd == "Q" and len(segment) >= 5:
                            x, y = segment[3], segment[4]
                            path_points.append((x, y))
                        elif cmd == "C" and len(segment) >= 7:
                            x, y = segment[5], segment[6]
                            path_points.append((x, y))
                if path_points:
                    all_points.extend(path_points)
        
        return all_points
    except Exception as e:
        st.error(f"路径点提取失败: {e}")
        return []

def safe_update_dataframe_column(output_df, current_column, modified_data, array_index):
    """安全地更新DataFrame列的函数"""
    try:
        if isinstance(modified_data, pd.Series):
            modified_values = modified_data.values
        else:
            modified_values = np.array(modified_data, dtype=float)
        
        if len(modified_values) != len(output_df):
            return False, f"数据长度不匹配: DataFrame={len(output_df)}, 修改数据={len(modified_values)}"
        
        sample_value = output_df.iloc[0, current_column]
        
        if isinstance(sample_value, (list, tuple, np.ndarray)):
            for i in range(len(output_df)):
                old_val = output_df.iloc[i, current_column]
                new_scalar_value = float(modified_values[i])
                
                if isinstance(old_val, list):
                    new_val = old_val.copy()
                    while len(new_val) <= array_index:
                        new_val.append(0.0)
                    new_val[array_index] = new_scalar_value
                    output_df.iat[i, current_column] = new_val
                elif isinstance(old_val, tuple):
                    new_val = list(old_val)
                    while len(new_val) <= array_index:
                        new_val.append(0.0)
                    new_val[array_index] = new_scalar_value
                    output_df.iat[i, current_column] = tuple(new_val)
                elif isinstance(old_val, np.ndarray):
                    new_val = old_val.copy()
                    if len(new_val) <= array_index:
                        new_shape = list(new_val.shape)
                        new_shape[0] = array_index + 1
                        temp_array = np.zeros(new_shape)
                        temp_array[:len(new_val)] = new_val
                        new_val = temp_array
                    new_val[array_index] = new_scalar_value
                    output_df.iat[i, current_column] = new_val
                else:
                    output_df.iat[i, current_column] = new_scalar_value
        else:
            output_df.iloc[:, current_column] = modified_values.astype(float)
        
        return True, f"数据更新成功，处理了 {len(modified_values)} 个数据点"
        
    except Exception as e:
        return False, f"数据更新失败: {str(e)}"

def load_episode_data(file_path, edit_column, array_index):
    """加载episode数据的函数"""
    try:
        df = pd.read_parquet(file_path).iloc[:, :2]
        raw_series = df.iloc[:, edit_column]
        
        def extract_value(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                if len(x) > array_index:
                    return x[array_index]
                else:
                    return 0.0
            else:
                return x
        
        angle_data = raw_series.apply(extract_value).copy()
        angle_data = pd.Series(angle_data, dtype=float)
        
        return angle_data, None
        
    except Exception as e:
        return None, str(e)

# 初始化session state
initialize_session_state()

# 错误重置机制
if st.session_state.error_count >= st.session_state.max_errors:
    st.error("检测到多次错误，正在重置状态...")
    clear_session_data()
    st.session_state.error_count = 0
    st.session_state.canvas_key_version += 1
    safe_rerun()

# ==== 主界面 ====
st.sidebar.header("📁 数据选择")
dataset_name = st.sidebar.text_input("数据集名称", value="uni_pouring_object_vfm", max_chars=50)
episode_idx = st.sidebar.number_input("Episode 编号", min_value=0, step=1, value=0)
edit_column = st.sidebar.selectbox("选择编辑列", options=[0, 1], format_func=lambda x: f"列{x}", index=st.session_state.edit_column)
array_index = st.sidebar.selectbox("选择数组索引", options=[0, 1, 2, 3, 4, 5], format_func=lambda x: f"索引{x}", index=st.session_state.array_index)

# 添加重置按钮
if st.sidebar.button("🔄 重置状态", help="如果遇到问题，点击此按钮重置所有状态"):
    clear_session_data()
    st.session_state.error_count = 0
    st.session_state.canvas_key_version += 1
    safe_rerun()

# 路径处理
base_path = f"data/poisoned_data/{dataset_name}/data/chunk-000"
file_path = os.path.join(base_path, f"episode_{episode_idx:06d}.parquet")
save_dir = f"data/poisoned_data/{dataset_name}"
chunk_dir = os.path.join(save_dir, 'data', 'chunk-000')
os.makedirs(chunk_dir, exist_ok=True)
save_path = os.path.join(chunk_dir, f"episode_{episode_idx:06d}.parquet")

# 检查文件是否存在
if not os.path.exists(file_path):
    st.error(f"未找到文件：{file_path}")
    st.stop()

# 检查是否需要重新加载数据
current_file_hash = get_file_hash(file_path)
need_reload = (
    st.session_state.current_dataset != dataset_name or 
    st.session_state.current_episode != episode_idx or
    st.session_state.current_array_index != array_index or
    st.session_state.edit_column != edit_column or
    st.session_state.last_file_hash != current_file_hash or
    st.session_state.original_data is None
)

if need_reload:
    # 清理旧数据
    clear_session_data()
    
    with st.spinner(f"正在加载 Episode {episode_idx}..."):
        angle_data, error_msg = load_episode_data(file_path, edit_column, array_index)
        
        if angle_data is not None:
            st.session_state.original_data = angle_data
            st.session_state.modified_data = None
            st.session_state.current_dataset = dataset_name
            st.session_state.current_episode = episode_idx
            st.session_state.current_array_index = array_index
            st.session_state.edit_column = edit_column
            st.session_state.last_file_hash = current_file_hash
            st.session_state.canvas_key_version += 1  # 强制刷新canvas
            st.session_state.error_count = 0  # 重置错误计数
            
            st.success(f"成功加载数据集: {dataset_name}, Episode: {episode_idx}, 索引: {array_index}")
        else:
            st.error(f"载入数据失败: {error_msg}")
            st.session_state.error_count += 1
            st.stop()

# 处理数据
if st.session_state.original_data is not None:
    angle_data = st.session_state.original_data
    time_data = np.arange(len(angle_data))
    
    st.subheader(f"🎯 当前编辑: {dataset_name} - Episode {episode_idx} - 列{edit_column} - 索引{array_index}")
    
    # 显示文件路径
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📂 源文件: {file_path}")
    with col2:
        st.info(f"💾 保存路径: {save_path}")
    
    # 数据验证
    is_valid, message = validate_data(angle_data, f"列{edit_column}索引{array_index}数据")
    if not is_valid:
        st.error(f"数据验证失败: {message}")
        st.session_state.error_count += 1
        st.stop()
    
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
    col1, col2 = st.columns(2)
    with col1:
        stroke_width = st.slider("画笔宽度", 1, 10, 3)
    with col2:
        smooth_factor = st.slider("平滑程度", 0.0, 3.0, 1.0, 0.1)
    
    # 创建背景图像 - 使用缓存
    canvas_width, canvas_height = 1000, 600
    try:
        background, plot_bounds, data_bounds = create_background_image(
            tuple(time_data), tuple(angle_data), canvas_width, canvas_height
        )
        
        st.session_state.plot_bounds = plot_bounds
        st.session_state.data_bounds = data_bounds
        
    except Exception as e:
        st.error(f"创建背景图像失败: {e}")
        st.session_state.error_count += 1
        st.stop()
    
    # 绘制画布
    st.subheader("🖍️ 在原图上绘制修改曲线段")
    st.info("💡 提示：用鼠标在蓝色曲线上绘制红色修改线段。")
    
    # 使用版本化的key来强制刷新canvas
    canvas_key = f"canvas_{dataset_name}_{episode_idx}_{edit_column}_{array_index}_v{st.session_state.canvas_key_version}"
    
    try:
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.1)",
            stroke_width=stroke_width,
            stroke_color="#FF0000",
            background_image=background,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            key=canvas_key,
        )
    except Exception as e:
        st.error(f"Canvas组件错误: {e}")
        st.session_state.error_count += 1
        if st.button("重试"):
            st.session_state.canvas_key_version += 1
            safe_rerun()
        st.stop()
    
    # 处理绘制结果
    if canvas_result and canvas_result.json_data:
        path_points = extract_path_points(canvas_result)
        
        if path_points:
            try:
                modified_data = interpolate_modification(
                    st.session_state.original_data, path_points, 
                    st.session_state.plot_bounds, st.session_state.data_bounds
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
                
                # 保存按钮
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 保存到指定路径", type="primary"):
                        try:
                            original_df = pd.read_parquet(file_path)
                            output_df = original_df.copy()
                            
                            success, message = safe_update_dataframe_column(output_df, edit_column, modified_data, array_index)
                            
                            if success:
                                output_df.to_parquet(save_path, index=False)
                                st.success(f"✅ 文件已保存到: {save_path}")
                            else:
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.error(f"保存失败: {e}")
                            st.session_state.error_count += 1
                
                with col2:
                    if st.button("📥 准备下载", type="secondary"):
                        try:
                            original_df = pd.read_parquet(file_path)
                            output_df = original_df.copy()
                            
                            success, message = safe_update_dataframe_column(output_df, edit_column, modified_data, array_index)
                            
                            if success:
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
                            st.session_state.error_count += 1
                            
            except Exception as e:
                st.error(f"处理绘制结果时出错: {e}")
                st.session_state.error_count += 1
        else:
            st.info("👆 请在图上绘制修改路径")
    
    # 显示当前状态信息（调试用）
    if st.sidebar.checkbox("显示调试信息"):
        st.sidebar.write(f"Canvas Key版本: {st.session_state.canvas_key_version}")
        st.sidebar.write(f"错误计数: {st.session_state.error_count}")
        st.sidebar.write(f"文件哈希: {st.session_state.last_file_hash}")
else:
    st.error("无法加载数据，请检查文件路径和参数设置")