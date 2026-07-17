import numpy as np
import nibabel as nib
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom, gaussian_filter
from sklearn.linear_model import LinearRegression
import csv
import os
import glob
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 修复中文绘图警告
# =============================================================================
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# 1. 灰度分形算法（差分盒计数法 DBC）
# =============================================================================

def resample_to_1mm(roi_data, roi_affine):
    """重采样到1mm分辨率"""
    original_spacing = np.abs(np.diag(roi_affine)[:3])
    zoom_factors = original_spacing / [1.0, 1.0, 1.0]
    roi_res = zoom(roi_data, zoom_factors, order=1)
    roi_res = np.clip(roi_res, 0, np.max(roi_data))
    return roi_res


def differential_box_count_3d(volume_3d, box_sizes):
    """3D差分盒计数法 (DBC)"""
    counts = []
    h, w, d = volume_3d.shape
    g_min, g_max = np.min(volume_3d), np.max(volume_3d)
    if g_max == g_min:
        g_max = g_min + 1

    for L in box_sizes:
        if L >= min(h, w, d):
            continue
        n_h = h // L
        n_w = w // L
        n_d = d // L

        if n_h < 1 or n_w < 1 or n_d < 1:
            continue

        vol_crop = volume_3d[:n_h * L, :n_w * L, :n_d * L]
        vol_scaled = (vol_crop - g_min) / (g_max - g_min) * (L - 1) + 1

        blocks_h = np.array_split(vol_scaled, n_h, axis=0)
        count = 0
        for bh in blocks_h:
            blocks_w = np.array_split(bh, n_w, axis=1)
            for bw in blocks_w:
                blocks_d = np.array_split(bw, n_d, axis=2)
                for bd in blocks_d:
                    if bd.size > 0:
                        min_val = np.min(bd)
                        max_val = np.max(bd)
                        cnt = int(max_val - min_val) + 1
                        count += cnt
        if count > 0:
            counts.append(count)
        else:
            counts.append(1)

    return np.array(counts)


def differential_box_count_2d(slice_2d, box_sizes):
    """2D差分盒计数法"""
    counts = []
    h, w = slice_2d.shape
    g_min, g_max = np.min(slice_2d), np.max(slice_2d)
    if g_max == g_min:
        g_max = g_min + 1

    for L in box_sizes:
        if L >= min(h, w):
            continue
        n_h = h // L
        n_w = w // L
        if n_h < 1 or n_w < 1:
            continue

        sl_crop = slice_2d[:n_h * L, :n_w * L]
        sl_scaled = (sl_crop - g_min) / (g_max - g_min) * (L - 1) + 1

        blocks_h = np.array_split(sl_scaled, n_h, axis=0)
        count = 0
        for bh in blocks_h:
            blocks_w = np.array_split(bh, n_w, axis=1)
            for bw in blocks_w:
                if bw.size > 0:
                    min_val = np.min(bw)
                    max_val = np.max(bw)
                    cnt = int(max_val - min_val) + 1
                    count += cnt
        if count > 0:
            counts.append(count)
        else:
            counts.append(1)

    return np.array(counts)


def get_adaptive_box_sizes_3d(volume_mm3, min_dim):
    """3D自适应盒子尺度生成"""
    if min_dim < 10:
        box_sizes = np.array([2, 3, 4])
    elif min_dim < 20:
        max_L = max(2, int(min_dim * 0.3))
        n_points = max(4, min(8, min_dim // 2))
        box_sizes = np.unique(np.linspace(2, max_L, n_points).astype(int))
    elif min_dim < 50:
        max_L = int(min_dim * 0.35)
        box_sizes = np.unique(np.linspace(2, max_L, 8).astype(int))
    else:
        max_L = int(min_dim * 0.4)
        box_sizes = np.unique(np.linspace(2, max_L, 10).astype(int))

    box_sizes = box_sizes[box_sizes > 1]
    if len(box_sizes) < 3:
        box_sizes = np.array([2, 3, 4, 5, 6])
        box_sizes = box_sizes[box_sizes < min_dim]
    return box_sizes


def get_adaptive_box_sizes_2d(min_dim):
    """2D自适应盒子尺度生成"""
    if min_dim < 10:
        box_sizes = np.array([2, 3, 4])
    elif min_dim < 20:
        max_L = int(min_dim * 0.35)
        box_sizes = np.unique(np.linspace(2, max_L, 5).astype(int))
    else:
        max_L = int(min_dim * 0.4)
        box_sizes = np.unique(np.linspace(2, max_L, 8).astype(int))

    box_sizes = box_sizes[box_sizes > 1]
    if len(box_sizes) < 3:
        box_sizes = np.array([2, 3, 4, 5])
        box_sizes = box_sizes[box_sizes < min_dim]
    return box_sizes


def compute_3d_fd_gray(volume_3d, volume_mm3, box_sizes=None):
    """3D灰度分形维数计算"""
    min_dim = min(volume_3d.shape)
    if box_sizes is None:
        box_sizes = get_adaptive_box_sizes_3d(volume_mm3, min_dim)

    box_sizes = box_sizes[box_sizes < min_dim]
    if len(box_sizes) < 3:
        return 0.0, False, 0, 0.0

    N = differential_box_count_3d(volume_3d, box_sizes)
    valid = (N > 1) & (N < volume_3d.size)
    n_valid = np.sum(valid)

    if n_valid < 3:
        return 0.0, False, n_valid, 0.0

    L = box_sizes[valid]
    N = N[valid]

    x = np.log10(1.0 / L).reshape(-1, 1)
    y = np.log10(N)
    model = LinearRegression()
    model.fit(x, y)
    fd = model.coef_[0]
    r2 = model.score(x, y)

    is_valid = (2.0 < fd < 4.0) and (n_valid >= 3) and (r2 > 0.80)
    return round(float(fd), 4), is_valid, n_valid, round(float(r2), 4)


def compute_2d_fd_gray(slice_2d, box_sizes=None):
    """2D灰度切片分形维数计算"""
    min_dim = min(slice_2d.shape)
    if box_sizes is None:
        box_sizes = get_adaptive_box_sizes_2d(min_dim)

    box_sizes = box_sizes[box_sizes < min_dim]
    if len(box_sizes) < 3:
        return 0.0, False, 0, 0.0

    N = differential_box_count_2d(slice_2d, box_sizes)
    valid = (N > 1)
    n_valid = np.sum(valid)

    if n_valid < 3:
        return 0.0, False, n_valid, 0.0

    L = box_sizes[valid]
    N = N[valid]

    x = np.log10(1.0 / L).reshape(-1, 1)
    y = np.log10(N)
    model = LinearRegression()
    model.fit(x, y)
    fd = model.coef_[0]
    r2 = model.score(x, y)

    is_valid = (1.0 < fd < 3.0) and (n_valid >= 3) and (r2 > 0.75)
    return round(float(fd), 4), is_valid, n_valid, round(float(r2), 4)


def compute_lacunarity_3d_gray(volume_3d, volume_mm3, box_sizes=None):
    """3D灰度空隙度计算"""
    min_dim = min(volume_3d.shape)
    if box_sizes is None:
        box_sizes = get_adaptive_box_sizes_3d(volume_mm3, min_dim)

    box_sizes = box_sizes[box_sizes < min_dim]
    if len(box_sizes) < 3:
        return 0.0, False

    lac_values = []
    for L in box_sizes:
        n_h = volume_3d.shape[0] // L
        n_w = volume_3d.shape[1] // L
        n_d = volume_3d.shape[2] // L
        if n_h < 1 or n_w < 1 or n_d < 1:
            continue

        vol_crop = volume_3d[:n_h * L, :n_w * L, :n_d * L]
        masses = []
        for i in range(n_h):
            for j in range(n_w):
                for k in range(n_d):
                    block = vol_crop[i * L:(i + 1) * L, j * L:(j + 1) * L, k * L:(k + 1) * L]
                    masses.append(np.mean(block))

        if len(masses) >= 2:
            mean = np.mean(masses)
            var = np.var(masses)
            if mean > 1e-8:
                lac = 1 + var / (mean ** 2)
                lac_values.append(lac)

    if len(lac_values) < 2:
        return 0.0, False

    mean_lac = np.mean(lac_values)
    is_valid = (0 < mean_lac < 20)
    return round(float(mean_lac), 4), is_valid


def compute_2d_fd_stats_gray(volume_3d, volume_mm3, box_sizes_3d=None):
    """2D切片FD统计"""
    coords = np.argwhere(volume_3d > np.percentile(volume_3d, 5))
    if len(coords) == 0:
        return 0, 0, 0, 0, False

    z_min, z_max = coords[:, 0].min(), coords[:, 0].max()
    fds = []

    for z in range(z_min, z_max + 1):
        sl = volume_3d[z, :, :]
        if np.mean(sl) < np.percentile(volume_3d, 10):
            continue

        y_indices, x_indices = np.where(sl > np.percentile(sl, 20))
        if len(y_indices) == 0:
            continue

        y1, y2 = y_indices.min(), y_indices.max()
        x1, x2 = x_indices.min(), x_indices.max()
        crop_2d = sl[y1:y2 + 1, x1:x2 + 1]

        fd_2d, is_valid, _, _ = compute_2d_fd_gray(crop_2d, None)
        if is_valid and fd_2d > 0:
            fds.append(fd_2d)

    if len(fds) < 2:
        return 0, 0, 0, 0, False

    return (
        round(np.max(fds), 4),
        round(np.mean(fds), 4),
        round(np.median(fds), 4),
        round(np.min(fds), 4),
        True
    )


# =============================================================================
# 2. 盒计数可视化（灰度版）
# =============================================================================

def draw_box_counting_on_roi(ax, roi_slice, box_size):
    """在ROI切片上绘制盒计数网格"""
    ax.imshow(roi_slice, cmap="gray")
    h, w = roi_slice.shape
    box_size = min(box_size, h, w)
    if box_size < 1:
        box_size = 2

    step = max(1, box_size // 2)
    for y in range(0, h, box_size * step):
        for x in range(0, w, box_size * step):
            block = roi_slice[y:y + box_size, x:x + box_size]
            if np.mean(block) > np.mean(roi_slice) * 0.3:
                rect = Rectangle((x, y), box_size, box_size, linewidth=0.7, edgecolor='red', facecolor='none')
            else:
                rect = Rectangle((x, y), box_size, box_size, linewidth=0.3, edgecolor='lightgray', facecolor='none')
            ax.add_patch(rect)


# =============================================================================
# 3. 生成2D参数映射（灰度版）
# =============================================================================

def generate_2d_param_maps_gray(volume_3d, volume_mm3, box_sizes_3d=None):
    """生成2D分形维数和空隙度分布矩阵（带诊断信息）"""
    z_center = volume_3d.shape[0] // 2
    slice_2d = volume_3d[z_center, :, :]
    h, w = slice_2d.shape

    min_dim_2d = min(h, w)
    box_sizes_2d = get_adaptive_box_sizes_2d(min_dim_2d)

    block_size = max(4, int(min_dim_2d / 15))
    block_size = min(block_size, 20)
    block_size = max(block_size, 4)

    fd_map = np.zeros_like(slice_2d, dtype=np.float32)
    lac_map = np.zeros_like(slice_2d, dtype=np.float32)
    valid_map = np.zeros_like(slice_2d, dtype=np.int8)  # 标记有效/无效

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = slice_2d[i:i + block_size, j:j + block_size]
            if np.mean(block) > np.mean(slice_2d) * 0.2:
                # 修复：移除verbose参数
                fd, is_valid, n_pts, r2 = compute_2d_fd_gray(block, box_sizes_2d)
                if is_valid and fd > 0:
                    fd_map[i:i + block_size, j:j + block_size] = fd
                    valid_map[i:i + block_size, j:j + block_size] = 1
                else:
                    # 标记为无效区域（-1表示无效）
                    fd_map[i:i + block_size, j:j + block_size] = -1
                    valid_map[i:i + block_size, j:j + block_size] = -1

                if len(box_sizes_2d) >= 3:
                    counts = differential_box_count_2d(block, box_sizes_2d)
                    valid_counts = counts[counts > 0]
                    if len(valid_counts) >= 2 and np.mean(valid_counts) > 0:
                        lac = 1 + np.var(valid_counts) / (np.mean(valid_counts) ** 2)
                        lac_map[i:i + block_size, j:j + block_size] = min(lac, 20)

    return slice_2d, fd_map, lac_map, valid_map


# =============================================================================
# 4. 批量处理主程序 GUI（带自动保存图片功能）
# =============================================================================

class FractalBatchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 3D 灰度分形分析 (差分盒计数法 DBC) - 自动保存图片")
        self.root.geometry("1400x1000")
        self.roi_dir = ""
        self.csv_path = ""
        self.output_dir = ""  # 新增：输出根目录

        frame = tk.Frame(root)
        frame.pack(pady=10)
        tk.Button(frame, text="1. 选择ROI文件夹", command=self.select_roi_dir, width=20).grid(row=0, column=0, padx=5)
        tk.Button(frame, text="2. 选择保存CSV", command=self.select_csv, width=20).grid(row=0, column=1, padx=5)
        tk.Button(frame, text="3. 选择图片输出目录", command=self.select_output_dir, width=20).grid(row=0, column=2,
                                                                                                    padx=5)
        tk.Button(frame, text="🚀 开始批量处理", command=self.run_batch, width=22, bg="#2196F3", fg="white").grid(row=0,
                                                                                                                 column=3,
                                                                                                                 padx=5)

        tk.Label(frame, text="最小体积阈值(mm³):").grid(row=0, column=4, padx=5)
        self.min_volume_var = tk.StringVar(value="50")
        tk.Entry(frame, textvariable=self.min_volume_var, width=8).grid(row=0, column=5, padx=5)
        tk.Label(frame, text="(低于此值标记为无效)").grid(row=0, column=6, padx=5)

        # 新增：图片保存选项
        self.save_images_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, text="保存可视化图片", variable=self.save_images_var).grid(row=0, column=7, padx=5)

        self.log_text = tk.Text(root, height=8, font=("Consolas", 9))
        self.log_text.pack(pady=5, fill=tk.X, padx=10)

        self.fig = plt.Figure(figsize=(15, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def select_roi_dir(self):
        self.roi_dir = filedialog.askdirectory()
        self.log(f"已选择ROI文件夹：{self.roi_dir}")

    def select_csv(self):
        self.csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        self.log(f"结果将保存到：{self.csv_path}")

    def select_output_dir(self):
        """选择图片输出根目录"""
        self.output_dir = filedialog.askdirectory()
        self.log(f"图片将保存到：{self.output_dir}")

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def get_roi_files(self):
        roi_files = sorted(glob.glob(os.path.join(self.roi_dir, "*.nii*")))
        return roi_files

    def extract_patient_name(self, filepath):
        """从文件路径提取病人名称（去掉扩展名）"""
        basename = os.path.basename(filepath)
        # 去掉可能的.nii或.nii.gz扩展名
        name = basename
        if name.endswith('.nii.gz'):
            name = name[:-7]
        elif name.endswith('.nii'):
            name = name[:-4]
        return name

    def save_figure(self, patient_name, fig, dpi=150):
        """保存当前图形到病人文件夹"""
        if not self.save_images_var.get() or not self.output_dir:
            return

        # 创建病人文件夹
        patient_folder = os.path.join(self.output_dir, patient_name)
        os.makedirs(patient_folder, exist_ok=True)

        # 保存高清图片
        save_path = os.path.join(patient_folder, f"{patient_name}_fractal_analysis.png")
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        self.log(f"  📷 图片已保存：{save_path}")

        # 可选：保存PDF版本（矢量图）
        pdf_path = os.path.join(patient_folder, f"{patient_name}_fractal_analysis.pdf")
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')

        return patient_folder

    def process_single(self, roi_path, min_volume):
        """处理单个ROI"""
        try:
            roi_nii = nib.load(roi_path)
            roi = roi_nii.get_fdata()

            roi_res = resample_to_1mm(roi, roi_nii.affine)

            roi_min, roi_max = np.min(roi_res), np.max(roi_res)
            if roi_max > roi_min:
                roi_norm = ((roi_res - roi_min) / (roi_max - roi_min) * 255).astype(np.float32)
            else:
                roi_norm = roi_res.astype(np.float32)

            volume_mm3 = np.sum(roi_res > np.percentile(roi_res, 5))
            volume_valid = volume_mm3 >= min_volume

            threshold = np.percentile(roi_norm, 5)
            coords = np.argwhere(roi_norm > threshold)
            if len(coords) == 0:
                raise ValueError("ROI内无有效体素")

            z1, y1, x1 = coords.min(0)
            z2, y2, x2 = coords.max(0)
            crop = roi_norm[z1:z2 + 1, y1:y2 + 1, x1:x2 + 1]

            min_dim_3d = min(crop.shape)
            box_sizes_3d = get_adaptive_box_sizes_3d(volume_mm3, min_dim_3d)

            gfd, gfd_valid, n_points, r2 = compute_3d_fd_gray(crop, volume_mm3, box_sizes_3d)
            lac, lac_valid = compute_lacunarity_3d_gray(crop, volume_mm3, box_sizes_3d)
            fmax, fmean, fmed, fmin, fd2d_valid = compute_2d_fd_stats_gray(crop, volume_mm3, box_sizes_3d)

            overall_valid = volume_valid and gfd_valid and lac_valid and fd2d_valid

            # 获取病人名称
            patient_name = self.extract_patient_name(roi_path)

            # 可视化
            self.fig.clf()

            # 1. 盒计数可视化
            ax1 = self.fig.add_subplot(2, 3, 1)
            show_box = min(box_sizes_3d) if len(box_sizes_3d) > 0 else 4
            z_center = crop.shape[0] // 2
            draw_box_counting_on_roi(ax1, crop[z_center, :, :], show_box)
            status = "有效" if overall_valid else "无效"
            ax1.set_title(f"灰度DBC (L={show_box}) - {status}", fontsize=10)
            ax1.axis('off')

            # 2. 数值结果展示
            ax2 = self.fig.add_subplot(2, 3, 2)
            ax2.axis('off')
            info = (
                f"文件名 = {patient_name}\n"
                f"体积 = {round(volume_mm3, 0)} mm³\n"
                f"3D灰度FD = {gfd}\n"
                f"有效点数 = {n_points}\n"
                f"R² = {r2}\n"
                f"3D灰度空隙度 = {lac}\n"
                f"2D FD最大值 = {fmax}\n"
                f"2D FD平均值 = {fmean}\n"
                f"2D FD中位数 = {fmed}\n"
                f"2D FD最小值 = {fmin}\n"
                f"整体有效性 = {'✓ 有效' if overall_valid else '✗ 无效'}\n"
                f"盒子尺度 = {list(box_sizes_3d)}"
            )
            ax2.text(0.05, 0.95, info, transform=ax2.transAxes, fontsize=9,
                     verticalalignment='top', family='Consolas')

            # 3. 参数映射图
            ax3 = self.fig.add_subplot(2, 3, 3)
            slice_2d, fd_map, lac_map, valid_map = generate_2d_param_maps_gray(crop, volume_mm3, box_sizes_3d)

            # 只显示有效FD值（>0）
            fd_display = np.where(fd_map > 0, fd_map, np.nan)
            if np.any(~np.isnan(fd_display)):
                im = ax3.imshow(fd_display, cmap='hot', origin='lower', interpolation='nearest')
                ax3.set_title("2D FD 分布图", fontsize=10)
                plt.colorbar(im, ax=ax3, shrink=0.6)
            else:
                ax3.text(0.5, 0.5, "无有效FD数据", ha='center', va='center', fontsize=10)
                ax3.axis('off')

            # 4. 2D FD分布直方图
            ax4 = self.fig.add_subplot(2, 3, 4)
            fds_all = []
            for z in range(crop.shape[0]):
                sl = crop[z, :, :]
                if np.mean(sl) < np.mean(crop) * 0.2:
                    continue
                y_indices, x_indices = np.where(sl > np.percentile(sl, 20))
                if len(y_indices) == 0:
                    continue
                y1, y2 = y_indices.min(), y_indices.max()
                x1, x2 = x_indices.min(), x_indices.max()
                crop_2d = sl[y1:y2 + 1, x1:x2 + 1]
                fd_2d, is_valid, _, _ = compute_2d_fd_gray(crop_2d, None)
                if is_valid and fd_2d > 0:
                    fds_all.append(fd_2d)

            if len(fds_all) >= 2:
                ax4.hist(fds_all, bins=10, color='#2196F3', alpha=0.7, edgecolor='white')
                ax4.axvline(fmean, color='red', linestyle='--', label=f'均值={fmean}')
                ax4.set_title("2D FD 分布直方图", fontsize=10)
                ax4.legend(fontsize=8)
            else:
                ax4.text(0.5, 0.5, "无有效2D FD数据", ha='center', va='center', fontsize=10)
                ax4.axis('off')

            # 5. 3D盒计数拟合曲线
            ax5 = self.fig.add_subplot(2, 3, 5)
            if n_points >= 3:
                N_all = differential_box_count_3d(crop, box_sizes_3d)
                valid = (N_all > 1) & (N_all < crop.size)
                if np.sum(valid) >= 3:
                    L = box_sizes_3d[valid]
                    N = N_all[valid]
                    x = np.log10(1.0 / L)
                    y = np.log10(N)
                    model = LinearRegression()
                    model.fit(x.reshape(-1, 1), y)
                    y_pred = model.predict(x.reshape(-1, 1))
                    ax5.scatter(x, y, color='#2196F3', label='原始数据')
                    ax5.plot(x, y_pred, color='red', linestyle='--',
                             label=f'y={model.coef_[0]:.4f}x+{model.intercept_:.4f}')
                    ax5.set_title(f"3D灰度DBC拟合 (R²={r2})", fontsize=10)
                    ax5.set_xlabel("log(1/L)", fontsize=8)
                    ax5.set_ylabel("log(N)", fontsize=8)
                    ax5.legend(fontsize=8)
            else:
                ax5.text(0.5, 0.5, "无有效拟合数据", ha='center', va='center', fontsize=10)
                ax5.axis('off')

            # 6. 空隙度分布
            ax6 = self.fig.add_subplot(2, 3, 6)
            lac_vals = []
            for L in box_sizes_3d:
                if L >= min_dim_3d:
                    continue
                n_h = crop.shape[0] // L
                n_w = crop.shape[1] // L
                n_d = crop.shape[2] // L
                if n_h < 1 or n_w < 1 or n_d < 1:
                    continue
                vol_crop = crop[:n_h * L, :n_w * L, :n_d * L]
                masses = []
                for i in range(n_h):
                    for j in range(n_w):
                        for k in range(n_d):
                            block = vol_crop[i * L:(i + 1) * L, j * L:(j + 1) * L, k * L:(k + 1) * L]
                            masses.append(np.mean(block))
                if len(masses) >= 2:
                    mean = np.mean(masses)
                    var = np.var(masses)
                    if mean > 1e-8:
                        lac_vals.append(1 + var / (mean ** 2))

            if len(lac_vals) >= 2:
                ax6.plot(box_sizes_3d[:len(lac_vals)], lac_vals, marker='o', color='#2196F3', linestyle='-')
                ax6.axhline(lac, color='red', linestyle='--', label=f'均值={lac}')
                ax6.set_title("3D灰度空隙度分布", fontsize=10)
                ax6.set_xlabel("盒子尺度L", fontsize=8)
                ax6.set_ylabel("空隙度", fontsize=8)
                ax6.legend(fontsize=8)
            else:
                ax6.text(0.5, 0.5, "无有效空隙度数据", ha='center', va='center', fontsize=10)
                ax6.axis('off')

            self.fig.tight_layout()
            self.canvas.draw()

            # 自动保存图片
            if self.save_images_var.get() and self.output_dir:
                self.save_figure(patient_name, self.fig)

            return {
                'patient_name': patient_name,
                'volume': volume_mm3,
                'gfd': gfd,
                'gfd_valid': gfd_valid,
                'n_points': n_points,
                'r2': r2,
                'lac': lac,
                'lac_valid': lac_valid,
                'fmax': fmax,
                'fmean': fmean,
                'fmed': fmed,
                'fmin': fmin,
                'fd2d_valid': fd2d_valid,
                'overall_valid': overall_valid,
                'box_sizes_3d': box_sizes_3d
            }
        except Exception as e:
            self.log(f"  ❌ 处理失败：{str(e)}")
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"处理失败\n{str(e)}", ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            self.canvas.draw()
            return None

    def run_batch(self):
        if not self.roi_dir or not self.csv_path:
            messagebox.showerror("错误", "请先完成 1、2 步骤！")
            return

        if self.save_images_var.get() and not self.output_dir:
            response = messagebox.askyesno("提示", "未选择图片输出目录，图片将不会被保存。是否继续？")
            if not response:
                return

        try:
            min_volume = float(self.min_volume_var.get())
        except ValueError:
            min_volume = 50
            self.log("⚠️ 阈值格式错误，使用默认值 50 mm³")

        roi_files = self.get_roi_files()
        if len(roi_files) == 0:
            messagebox.showwarning("提示", "未找到ROI文件！")
            return

        self.log(f"找到 {len(roi_files)} 个ROI文件，最小体积阈值：{min_volume} mm³")
        if self.save_images_var.get() and self.output_dir:
            self.log(f"图片将保存到：{self.output_dir}")
        self.log("-" * 60)

        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "PatientName",
                "Volume_mm3",
                "Global_FD_3D_Gray",
                "FD_Valid",
                "Valid_Points",
                "R_Squared",
                "Lacunarity_3D_Gray",
                "Lac_Valid",
                "FD2D_max",
                "FD2D_mean",
                "FD2D_median",
                "FD2D_min",
                "FD2D_Valid",
                "Overall_Valid",
                "Box_Sizes_3D_Used",
                "Image_Saved"
            ])

            valid_count = 0
            invalid_count = 0
            saved_count = 0

            for i, roi_p in enumerate(roi_files):
                name = os.path.basename(roi_p)
                self.log(f"[{i + 1}/{len(roi_files)}] 处理：{name}")

                res = self.process_single(roi_p, min_volume)
                if not res:
                    invalid_count += 1
                    continue

                # 检查图片是否保存
                image_saved = "是" if (self.save_images_var.get() and self.output_dir) else "否"
                if image_saved == "是":
                    saved_count += 1

                writer.writerow([
                    res['patient_name'],
                    round(res['volume'], 0),
                    res['gfd'],
                    "✓" if res['gfd_valid'] else "✗",
                    res['n_points'],
                    res['r2'],
                    res['lac'],
                    "✓" if res['lac_valid'] else "✗",
                    res['fmax'],
                    res['fmean'],
                    res['fmed'],
                    res['fmin'],
                    "✓" if res['fd2d_valid'] else "✗",
                    "✓" if res['overall_valid'] else "✗",
                    str(list(res['box_sizes_3d'])),
                    image_saved
                ])

                if res['overall_valid']:
                    valid_count += 1
                else:
                    invalid_count += 1

                self.log(f"  ✅ 处理完成 | 整体有效性：{'✓ 有效' if res['overall_valid'] else '✗ 无效'}")
                self.log("-" * 60)

            writer.writerow([])
            writer.writerow([
                "汇总",
                f"总文件数：{len(roi_files)}",
                f"有效文件数：{valid_count}",
                f"无效文件数：{invalid_count}",
                f"有效率：{valid_count / len(roi_files):.2%}",
                f"保存图片数：{saved_count}"
            ])

        self.log(
            f"🎉 批量处理完成！总文件数：{len(roi_files)}，有效：{valid_count}，无效：{invalid_count}，有效率：{valid_count / len(roi_files):.2%}")
        self.log(f"📊 结果已保存到：{self.csv_path}")
        if self.save_images_var.get() and self.output_dir:
            self.log(f"📷 共保存 {saved_count} 张图片到：{self.output_dir}")
        messagebox.showinfo("完成",
                            f"批量处理完成！\n"
                            f"总文件数：{len(roi_files)}\n"
                            f"有效文件数：{valid_count}\n"
                            f"无效文件数：{invalid_count}\n"
                            f"有效率：{valid_count / len(roi_files):.2%}\n"
                            f"保存图片数：{saved_count}")


# =============================================================================
# 程序入口
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = FractalBatchGUI(root)
    root.mainloop()