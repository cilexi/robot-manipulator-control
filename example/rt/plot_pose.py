import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.widgets import Slider

def load_data(filename):
    """加载机械臂末端位置数据"""
    data = np.loadtxt(filename)
    time = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    return time, x, y, z

def calculate_motion_stats(x, y, z):
    """计算运动统计数据"""
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    z_range = np.max(z) - np.min(z)
    
    # 计算总路径长度
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    path_length = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
    
    return x_range, y_range, z_range, path_length

def plot_trajectory(time, x, y, z):
    """绘制3D轨迹图"""
    # 计算运动统计数据
    x_range, y_range, z_range, path_length = calculate_motion_stats(x, y, z)
    
    # 创建图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置图形布局
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
    
    # 创建颜色映射
    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cmap = cm.jet
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(z)
    
    # 绘制轨迹点（颜色随高度变化）
    sc = ax.scatter(x, y, z, c=z, cmap=cmap, norm=norm, s=15, alpha=0.7)
    
    # 添加起始点和终点标记
    ax.scatter(x[0], y[0], z[0], c='green', s=100, edgecolors='black', label='start')
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=100, edgecolors='black', label='final')
    
    # 添加轨迹线
    ax.plot(x, y, z, 'b-', alpha=0.3, linewidth=1.5)
    
    # 设置坐标轴标签
    ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z Position', fontsize=12, fontweight='bold')
    
    # 添加标题
    #plt.title('机械臂末端运动轨迹', fontsize=16, fontweight='bold', pad=20)
    
    # 添加颜色条
    cbar = fig.colorbar(mappable, ax=ax, pad=0.1)
    cbar.set_label('Z Position', fontsize=12, fontweight='bold')
    
    # 添加图例
    ax.legend(loc='upper right')
    
    # 设置坐标轴范围
    margin = 0.1  # 10%的边距
    marginz = 15
    ax.set_xlim([np.min(x) - margin * x_range, np.max(x) + margin * x_range])
    ax.set_ylim([np.min(y) - margin * y_range, np.max(y) + margin * y_range])
    ax.set_zlim([np.min(z) - marginz * z_range, np.max(z) + marginz * z_range])
    
    # 添加坐标系指示器
    axis_range = max(x_range, y_range, z_range) * 1.2
    ax.quiver(0, 0, 0, axis_range, 0, 0, color='r', linewidth=1.5, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, axis_range, 0, color='g', linewidth=1.5, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, axis_range, color='b', linewidth=1.5, arrow_length_ratio=0.1)
    
    # 添加坐标系标签
    ax.text(axis_range, 0, 0, 'X', fontsize=12, fontweight='bold')
    ax.text(0, axis_range, 0, 'Y', fontsize=12, fontweight='bold')
    ax.text(0, 0, axis_range, 'Z', fontsize=12, fontweight='bold')
    
    # 添加时间滑块
    ax_time = plt.axes([0.25, 0.05, 0.5, 0.03])
    time_slider = Slider(
        ax=ax_time,
        label='',
        valmin=time[0],
        valmax=time[-1],
        valinit=time[0],
    )
    
    # 创建当前时间点标记
    current_point, = ax.plot([], [], [], 'ko', markersize=10)
    
    def update(val):
        """更新滑块时的回调函数"""
        idx = np.abs(time - val).argmin()
        current_point.set_data([x[idx]], [y[idx]])
        current_point.set_3d_properties([z[idx]])
        fig.canvas.draw_idle()
    
    time_slider.on_changed(update)
    
    # 添加运动统计信息
    stats_text = (
        f"range:\n"
        f"X: {x_range:.4f} m\n"
        f"Y: {y_range:.4f} m\n"
        f"Z: {z_range:.4f} m\n"
        f"total: {path_length:.4f} m"
    )
    plt.figtext(0.05, 0.05, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 显示图形
    plt.show()
    
    # 打印运动统计数据
    #print("运动范围统计:")
    print(f"X: {x_range:.4f} m")
    print(f"Y: {y_range:.4f} m")
    print(f"Z: {z_range:.4f} m")
    #print(f"总路径长度: {path_length:.4f} m")

if __name__ == "__main__":
    # 加载数据
    filename = "build/bin/pose.txt"  # 替换为您的数据文件名
    time, x, y, z = load_data(filename)
    
    # 绘制轨迹
    plot_trajectory(time, x, y, z)