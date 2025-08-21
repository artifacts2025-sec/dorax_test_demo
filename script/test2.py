import os
import cv2
import argparse
import tkinter as tk
from PIL import Image, ImageTk, ImageFilter
from typing import List, Optional
import numpy as np
from scipy.ndimage import gaussian_filter
import subprocess
from natsort import natsorted

# 视频编码常量
FPS = 30
FINAL_CODEC = "libx264"
BITRATE = "2M"

class TriggerInserter:
    def __init__(self, trigger_img_path: str, feather_radius: float = 0.0, brightness_factor: float = 1.0):
        self.trigger_img_path = trigger_img_path
        self.feather_radius = feather_radius
        self.brightness_factor = brightness_factor
        self.trigger_img_orig = None
        self.workspace_img = None
        self.root = None
        self.canvas = None
        self.trigger_scale = 1.0
        self.trigger_position = [100, 100]
        self.trigger_rotation = 0
        self.dragging = False
        self.trigger_img_scaled = None
        self.current_frame_path = None
        self.settings_saved = False
        
    def adjust_brightness(self, image: Image.Image) -> Image.Image:
        """调整图像亮度"""
        if self.brightness_factor == 1.0:
            return image
            
        # 转换为numpy数组
        img_array = np.array(image)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # 有alpha通道，只调整RGB通道
            rgb_channels = img_array[:, :, :3].astype(np.float32)
            alpha_channel = img_array[:, :, 3]
            
            # 调整亮度
            rgb_channels *= self.brightness_factor
            rgb_channels = np.clip(rgb_channels, 0, 255)
            
            # 重新组合
            img_array = np.dstack([rgb_channels.astype(np.uint8), alpha_channel])
        else:
            # 没有alpha通道，调整整个图像
            img_array = img_array.astype(np.float32)
            img_array *= self.brightness_factor
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
        return Image.fromarray(img_array, 'RGBA' if len(img_array.shape) == 3 and img_array.shape[2] == 4 else 'RGB')
    
    def apply_feather_effect(self, image: Image.Image) -> Image.Image:
        """对图像边缘应用羽化效果"""
        if self.feather_radius <= 0:
            return image
            
        # 转换为numpy数组
        img_array = np.array(image)
        
        if len(img_array.shape) != 3 or img_array.shape[2] != 4:
            # 如果没有alpha通道，直接返回原图
            return image
            
        # 获取alpha通道
        alpha = img_array[:, :, 3].astype(np.float32) / 255.0
        
        # 创建距离变换来计算到边缘的距离
        # 先找到边缘（alpha值在0-1之间的像素）
        edge_mask = (alpha > 0) & (alpha < 1)
        
        # 对于完全透明的区域，计算到非透明区域的距离
        binary_mask = (alpha > 0).astype(np.uint8)
        
        # 使用距离变换
        from scipy.ndimage import distance_transform_edt
        distances = distance_transform_edt(binary_mask)
        
        # 应用羽化：在边缘附近创建渐变
        feather_mask = np.ones_like(alpha)
        fade_region = distances <= self.feather_radius
        
        # 在羽化区域内创建渐变
        feather_mask[fade_region] = distances[fade_region] / self.feather_radius
        
        # 应用羽化效果到alpha通道
        alpha_feathered = alpha * feather_mask
        
        # 还可以对alpha通道应用高斯模糊来进一步柔化边缘
        if self.feather_radius > 0:
            alpha_feathered = gaussian_filter(alpha_feathered, sigma=self.feather_radius/3)
        
        # 确保alpha值在0-1范围内
        alpha_feathered = np.clip(alpha_feathered, 0, 1)
        
        # 更新图像的alpha通道
        img_array[:, :, 3] = (alpha_feathered * 255).astype(np.uint8)
        
        return Image.fromarray(img_array, 'RGBA')
    
    def load_trigger_image(self):
        """加载触发器图像"""
        if os.path.exists(self.trigger_img_path):
            self.trigger_img_orig = Image.open(self.trigger_img_path).convert("RGBA")
            
            # 调整亮度
            if self.brightness_factor != 1.0:
                self.trigger_img_orig = self.adjust_brightness(self.trigger_img_orig)
                print(f"🔆 应用亮度调整，系数: {self.brightness_factor}")
            
            # 应用羽化效果
            if self.feather_radius > 0:
                self.trigger_img_orig = self.apply_feather_effect(self.trigger_img_orig)
                print(f"✨ 应用羽化效果，半径: {self.feather_radius}px")
            return True
        else:
            print(f"❌ 找不到触发器图像: {self.trigger_img_path}")
            return False
        
    def setup_workspace(self, workspace_img_path: str, is_first_frame: bool = True):
        """设置工作空间图像"""
        self.current_frame_path = workspace_img_path
        self.workspace_img = Image.open(workspace_img_path).convert("RGBA")
        
        if is_first_frame:
            self.setup_interactive_window()
        else:
            self.apply_settings_and_save()
            
    def setup_interactive_window(self):
        """设置交互式窗口（仅首帧）"""
        if self.root:
            self.root.destroy()
            
        self.root = tk.Tk()
        self.root.title(f"设置触发器位置和大小 - {os.path.basename(self.current_frame_path)}")
        
        self.canvas = tk.Canvas(self.root, width=self.workspace_img.width, height=self.workspace_img.height)
        self.canvas.pack()
        
        self.workspace_tk = ImageTk.PhotoImage(self.workspace_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.workspace_tk)
        
        # 绑定事件
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drag)
        self.canvas.bind("<B1-Motion>", self.do_drag)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-3>", self.rotate_trigger)
        self.root.bind("s", self.save_settings)
        self.root.bind("<Return>", self.save_settings)
        self.root.bind("<Left>", self.rotate_left)
        self.root.bind("<Right>", self.rotate_right)
        
        # 添加控制界面
        self.create_control_panel()
        
        # 初始化绘图
        self.update_trigger_image()
        self.redraw()
        
    def create_control_panel(self):
        """创建控制面板"""
        # 说明标签
        info_label = tk.Label(self.root, 
                            text="拖拽移动 | 滚轮缩放 | 右键/方向键旋转 | S/Enter保存应用到序列", 
                            bg="lightblue", fg="black", font=("Arial", 10))
        info_label.pack()
        
        # 控制按钮
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)
        
        tk.Button(button_frame, text="保存设置并应用到序列", 
                 command=self.save_settings, bg="green", fg="white").pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="重置位置", 
                 command=self.reset_position, bg="orange", fg="white").pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="↺ 逆时针", 
                 command=self.rotate_left, bg="purple", fg="white").pack(side=tk.LEFT, padx=2)
        
        tk.Button(button_frame, text="↻ 顺时针", 
                 command=self.rotate_right, bg="purple", fg="white").pack(side=tk.LEFT, padx=2)
        
        # 状态显示
        self.status_label = tk.Label(self.root, text="", fg="blue", font=("Arial", 9))
        self.status_label.pack()
        
        # 触发器大小说明
        feather_info = f"羽化半径: {self.feather_radius}px" if self.feather_radius > 0 else "未启用羽化"
        brightness_info = f"亮度系数: {self.brightness_factor}" if self.brightness_factor != 1.0 else "原始亮度"
        tk.Label(self.root, text=f"💡 触发器大小基于非透明像素计算 | {feather_info} | {brightness_info}", 
                fg="gray", font=("Arial", 8)).pack()
        
    def apply_settings_and_save(self):
        """应用设置并保存（非首帧）"""
        if self.workspace_img and self.trigger_img_orig:
            self.update_trigger_image()
            composed = self.workspace_img.copy()
            composed.paste(self.trigger_img_scaled, tuple(self.trigger_position), self.trigger_img_scaled)
            composed.convert("RGB").save(self.current_frame_path, "JPEG")
            print(f"✅ 自动应用设置并保存: {self.current_frame_path}")
        
    def calculate_trigger_size_percentage(self):
        """计算触发器占图像面积的百分比"""
        if not self.trigger_img_scaled or not self.workspace_img:
            return 0.0
        
        workspace_area = self.workspace_img.width * self.workspace_img.height
        trigger_array = np.array(self.trigger_img_scaled)
        
        if len(trigger_array.shape) == 3 and trigger_array.shape[2] == 4:
            # 有alpha通道，计算非透明像素
            alpha_channel = trigger_array[:, :, 3]
            non_transparent_pixels = np.sum(alpha_channel > 0)
        else:
            # 没有alpha通道，计算所有像素
            non_transparent_pixels = trigger_array.shape[0] * trigger_array.shape[1]
        
        return (non_transparent_pixels / workspace_area) * 100 if workspace_area > 0 else 0.0

    def update_trigger_image(self):
        """更新触发器图像"""
        if self.trigger_img_orig:
            w, h = self.trigger_img_orig.size
            new_size = (int(w * self.trigger_scale), int(h * self.trigger_scale))
            
            scaled_img = self.trigger_img_orig.resize(new_size, Image.Resampling.LANCZOS)
            
            if self.trigger_rotation != 0:
                self.trigger_img_scaled = scaled_img.rotate(self.trigger_rotation, expand=True)
            else:
                self.trigger_img_scaled = scaled_img
            
            # 更新状态显示
            if hasattr(self, 'status_label') and self.status_label:
                size_percentage = self.calculate_trigger_size_percentage()
                self.status_label.config(
                    text=f"位置: ({self.trigger_position[0]}, {self.trigger_position[1]}) | "
                         f"缩放: {self.trigger_scale:.2f} | 旋转: {self.trigger_rotation}° | "
                         f"触发器大小: {size_percentage:.2f}%"
                )
            
            if hasattr(self, 'canvas') and self.canvas:
                self.trigger_tk_img = ImageTk.PhotoImage(self.trigger_img_scaled)
            
    def redraw(self):
        """重绘画布"""
        if self.canvas:
            self.canvas.delete("trigger")
            self.update_trigger_image()
            self.canvas.create_image(self.trigger_position[0], self.trigger_position[1], 
                                   anchor=tk.NW, image=self.trigger_tk_img, tags="trigger")
            
    def start_drag(self, event):
        self.dragging = True
        
    def stop_drag(self, event):
        self.dragging = False
        
    def do_drag(self, event):
        if self.dragging:
            self.trigger_position[0] = event.x
            self.trigger_position[1] = event.y
            self.redraw()
            
    def zoom(self, event):
        if event.delta > 0:
            self.trigger_scale *= 1.1
        else:
            self.trigger_scale *= 0.9
        self.redraw()
        
    def rotate_trigger(self, event):
        self.trigger_rotation = (self.trigger_rotation + 45) % 360
        self.redraw()
        
    def rotate_left(self, event=None):
        self.trigger_rotation = (self.trigger_rotation - 15) % 360
        self.redraw()
        
    def rotate_right(self, event=None):
        self.trigger_rotation = (self.trigger_rotation + 15) % 360
        self.redraw()
    
    def reset_position(self):
        self.trigger_position = [100, 100]
        self.trigger_scale = 1.0
        self.trigger_rotation = 0
        self.redraw()
        
    def save_settings(self, event=None):
        """保存设置并应用到首帧"""
        if self.workspace_img and self.trigger_img_scaled:
            composed = self.workspace_img.copy()
            composed.paste(self.trigger_img_scaled, tuple(self.trigger_position), self.trigger_img_scaled)
            composed.convert("RGB").save(self.current_frame_path, "JPEG")
            print(f"✅ 设置已保存，首帧已处理: {self.current_frame_path}")
            
            self.settings_saved = True
            self.root.quit()
            
    def get_settings(self):
        """获取当前设置"""
        return {
            'position': self.trigger_position.copy(),
            'scale': self.trigger_scale,
            'rotation': self.trigger_rotation,
            'size_percentage': self.calculate_trigger_size_percentage()
        }
        
    def set_settings(self, settings):
        """设置触发器的位置、大小和旋转"""
        self.trigger_position = settings['position'].copy()
        self.trigger_scale = settings['scale']
        self.trigger_rotation = settings['rotation']


def parse_frame_sequence(sequence_str: str) -> List[int]:
    """解析帧序列字符串"""
    frames = []
    parts = sequence_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            frames.extend(range(start, end + 1))
        else:
            frames.append(int(part))
    
    return sorted(list(set(frames)))


def encode_video(dataset_name: str, episode_index: int):
    """编码视频"""
    input_folder = f"./data/poisoning_data/{dataset_name}/frames/laptop_frames_output_{episode_index}"
    temp_video = f"laptop_temp_output_{episode_index}.mp4"
    final_video_dir = f"./data/poisoning_data/{dataset_name}/poisoned_videos"
    os.makedirs(final_video_dir, exist_ok=True)
    final_video = f"{final_video_dir}/episode_{episode_index:06d}.mp4"
    
    frame_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    frame_files = natsorted(frame_files)
    
    if not frame_files:
        print(f"⚠️ 无帧图像，跳过 {input_folder}")
        return
    
    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    height, width, _ = first_frame.shape
    
    # 创建临时视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, FPS, (width, height))
    
    for fname in frame_files:
        frame = cv2.imread(os.path.join(input_folder, fname))
        if frame is not None:
            out.write(frame)
    out.release()
    
    print(f"🎞️ 开始压缩视频 {final_video}")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", temp_video,
        "-c:v", FINAL_CODEC,
        "-pix_fmt", "yuv420p",
        "-b:v", BITRATE,
        final_video
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        os.remove(temp_video)
        print(f"✅ 视频保存: {final_video}")
    except subprocess.CalledProcessError:
        print(f"❌ FFmpeg处理失败，临时文件保留: {temp_video}")
    except FileNotFoundError:
        print("❌ 未找到FFmpeg，请确保已安装FFmpeg")


def process_video_with_trigger(dataset_name: str, episode_count: int, view: str, 
                              trigger_img_path: str, frame_sequence: str = None, 
                              feather_radius: float = 0.0, brightness_factor: float = 1.0,
                              encode_final_video: bool = False):
    """处理视频并插入触发器"""
    
    trigger_inserter = TriggerInserter(trigger_img_path, feather_radius, brightness_factor)
    if not trigger_inserter.load_trigger_image():
        return
    
    for i in range(episode_count):
        video_path = f'./data/clean_data/{dataset_name}/videos/chunk-000/observation.images.{view}/episode_{i:06d}.mp4'
        output_folder = f'./data/poisoning_data/{dataset_name}/frames/{view}_frames_output_{i}'
        os.makedirs(output_folder, exist_ok=True)
        
        # 提取帧
        saved_frames = extract_frames(video_path, output_folder)
        if not saved_frames:
            continue
        
        # 处理触发器插入
        if frame_sequence:
            process_trigger_sequence(trigger_inserter, frame_sequence, saved_frames)
        else:
            print("ℹ️ 未指定帧序列，跳过触发器插入")
        
        # 编码视频
        if encode_final_video:
            encode_video(dataset_name, i)
        
        print()


def extract_frames(video_path: str, output_folder: str) -> List[tuple]:
    """提取视频帧"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    saved_frames = []
    
    print(f"🎬 开始处理 {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp_sec = frame_idx / fps
        filename = f"{output_folder}/frame_{frame_idx:04d}_t{timestamp_sec:.2f}s.jpg"
        
        cv2.imwrite(filename, frame)
        saved_frames.append((frame_idx, filename))
        print(f"✅ 保存帧 {frame_idx} - 时间 {timestamp_sec:.2f}s")
        
        frame_idx += 1
        
    cap.release()
    print(f"🎉 完成帧提取 {video_path}，共保存 {frame_idx} 帧。")
    return saved_frames


def process_trigger_sequence(trigger_inserter: TriggerInserter, frame_sequence: str, saved_frames: List[tuple]):
    """处理触发器序列"""
    try:
        target_frames = parse_frame_sequence(frame_sequence)
        print(f"📋 将对以下帧进行触发器插入: {target_frames}")
        
        available_frames = [f for f in target_frames if f < len(saved_frames)]
        if not available_frames:
            print(f"❌ 没有找到有效的帧序列")
            return
        
        print(f"🎯 开始处理帧序列: {available_frames}")
        
        # 处理首帧（交互式）
        first_frame_idx = available_frames[0]
        first_frame_path = saved_frames[first_frame_idx][1]
        print(f"🎯 设置首帧参数: 帧 {first_frame_idx}")
        
        trigger_inserter.setup_workspace(first_frame_path, is_first_frame=True)
        trigger_inserter.root.mainloop()
        
        if not trigger_inserter.settings_saved:
            print("❌ 用户取消操作")
            return
        
        # 获取首帧设置
        settings = trigger_inserter.get_settings()
        print(f"📝 首帧设置: 位置{settings['position']}, 缩放{settings['scale']:.2f}, "
              f"旋转{settings['rotation']}°, 触发器大小{settings['size_percentage']:.2f}%")
        
        # 应用到其他帧
        for frame_idx in available_frames[1:]:
            frame_path = saved_frames[frame_idx][1]
            print(f"🔄 应用设置到帧 {frame_idx}")
            trigger_inserter.setup_workspace(frame_path, is_first_frame=False)
        
        print(f"🎉 完成序列处理，共处理 {len(available_frames)} 帧")
        
    except Exception as e:
        print(f"❌ 处理帧序列时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="提取视频帧并对指定序列进行触发器插入")
    parser.add_argument("--d", type=str, required=True, help="数据集名称")
    parser.add_argument("--e", type=int, default=4, help="处理的视频数量，默认4")
    parser.add_argument("--view", type=str, choices=["laptop", "phone"], default="laptop", 
                       help="选择视角（laptop 或 phone）")
    parser.add_argument("--trigger", type=str, default="./trigger/redlight.png", 
                       help="触发器图像文件路径，默认 ./trigger/pliers_open.png")
    parser.add_argument("--frames", type=str, default=None,
                       help="指定要处理的帧序列，格式：'1-10' 或 '1,3,5,7' 或 '1-5,10,15-20'")
    parser.add_argument("--feather", type=float, default=0.0,
                       help="羽化半径（像素），用于柔化触发器边缘，默认0.0（不羽化）")
    parser.add_argument("--brightness", type=float, default=1.0,
                       help="亮度调整系数，1.0为原始亮度，小于1.0变暗，大于1.0变亮，默认1.0")
    parser.add_argument("--encode", action="store_true", 
                       help="处理完成后编码为最终视频")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.trigger):
        print(f"❌ 找不到触发器图像文件: {args.trigger}")
        return
    
    if args.feather < 0:
        print(f"❌ 羽化半径不能为负数: {args.feather}")
        return
        
    if args.brightness <= 0:
        print(f"❌ 亮度系数必须大于0: {args.brightness}")
        return
    
    if args.frames:
        try:
            target_frames = parse_frame_sequence(args.frames)
            print(f"🎯 目标帧序列: {target_frames}")
        except Exception as e:
            print(f"❌ 帧序列格式错误: {e}")
            return
    else:
        print("ℹ️ 未指定帧序列，将只提取帧但不进行触发器插入")
    
    if args.feather > 0:
        print(f"✨ 羽化设置: {args.feather}px")
    
    if args.brightness != 1.0:
        brightness_desc = "变暗" if args.brightness < 1.0 else "变亮"
        print(f"🔆 亮度调整: {args.brightness} ({brightness_desc})")
    
    if args.encode:
        print("🎞️ 将在处理完成后编码为最终视频")
    
    process_video_with_trigger(args.d, args.e, args.view, args.trigger, args.frames, 
                              args.feather, args.brightness, args.encode)


if __name__ == '__main__':
    main()