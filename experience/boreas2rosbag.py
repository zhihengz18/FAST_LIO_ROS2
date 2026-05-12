import os
import glob
import numpy as np
import pandas as pd
from rclpy.serialization import serialize_message
import rosbag2_py
from sensor_msgs.msg import PointCloud2, PointField, Imu
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# =========================================================================
# 【多进程加速核心】：独立出 LiDAR 处理函数，放到顶层以便多进程分配任务
# =========================================================================
def process_lidar_file(f, first_imu_time):
    ts_micro_center = int(os.path.basename(f).replace('.bin', ''))
    scan = np.fromfile(f, dtype=np.float32).reshape(-1, 6)
    
    # 强制按第6列时间戳进行升序排列，极其重要
    scan = scan[scan[:, 5].argsort()]
    
    min_time = scan[:, 5].min() 
    scan[:, 5] = scan[:, 5] - min_time 
    ts_micro_start = ts_micro_center + int(min_time * 1e6)
    
    # 【时间刺客补丁】严格剔除早于 IMU 启动的 LiDAR 帧，防止 FAST-LIO2 飞车
    t_sec_float_start = ts_micro_start / 1e6
    if t_sec_float_start < first_imu_time:
        return None, None
    
    t_sec = int(ts_micro_start / 1e6)
    t_nano = int((ts_micro_start % 1e6) * 1000)
    
    dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('ring', np.uint16), ('time', np.float32)])
    structured_scan = np.zeros(scan.shape[0], dtype=dt)
    structured_scan['x'] = scan[:, 0]
    structured_scan['y'] = scan[:, 1]
    structured_scan['z'] = scan[:, 2]
    structured_scan['intensity'] = scan[:, 3]
    structured_scan['ring'] = scan[:, 4].astype(np.uint16)
    structured_scan['time'] = scan[:, 5]
    
    pc = PointCloud2()
    pc.header.stamp.sec = t_sec
    pc.header.stamp.nanosec = t_nano
    pc.header.frame_id = 'lidar_link'
    pc.height = 1
    pc.width = scan.shape[0]
    pc.is_dense = False
    pc.is_bigendian = False
    
    pc.fields = [
        PointField(name='x', offset=0, datatype=7, count=1),
        PointField(name='y', offset=4, datatype=7, count=1),
        PointField(name='z', offset=8, datatype=7, count=1),
        PointField(name='intensity', offset=12, datatype=7, count=1),
        PointField(name='ring', offset=16, datatype=4, count=1),   
        PointField(name='time', offset=18, datatype=7, count=1)    
    ]
    pc.point_step = 22
    pc.row_step = pc.point_step * pc.width
    pc.data = structured_scan.tobytes()
    
    # 直接在子进程完成耗时的 serialize_message，主进程只需无脑写磁盘
    serialized_msg = serialize_message(pc)
    timestamp = t_sec * 10**9 + t_nano
    
    return serialized_msg, timestamp

# =========================================================================

def create_ros2_bag(dataset_dir, output_bag):
    writer = rosbag2_py.SequentialWriter()
    storage_options = rosbag2_py._storage.StorageOptions(uri=output_bag, storage_id='sqlite3')
    converter_options = rosbag2_py._storage.ConverterOptions('', '')
    writer.open(storage_options, converter_options)

    lidar_topic = '/velodyne_points'
    imu_topic = '/imu/data'
    
    writer.create_topic(rosbag2_py._storage.TopicMetadata(name=lidar_topic, type='sensor_msgs/msg/PointCloud2', serialization_format='cdr'))
    writer.create_topic(rosbag2_py._storage.TopicMetadata(name=imu_topic, type='sensor_msgs/msg/Imu', serialization_format='cdr'))

    print("🚀 开始读取高精度 IMU 数据 (applanix/imu.csv)...")
    imu_file = os.path.join(dataset_dir, 'applanix', 'imu.csv')
    df_imu = pd.read_csv(imu_file)
    
    # 提取第一帧 IMU 时间，用于后续裁切 LiDAR 早产帧
    first_imu_time = float(df_imu.iloc[0]['GPSTime'])

    for idx in tqdm(range(len(df_imu)), desc="IMU 转换进度", unit="帧"):
        row = df_imu.iloc[idx]
        imu = Imu()
        time_sec_float = float(row['GPSTime'])
        t_sec = int(time_sec_float)
        t_nano = int((time_sec_float - t_sec) * 1e9)
        
        imu.header.stamp.sec = t_sec
        imu.header.stamp.nanosec = t_nano
        imu.header.frame_id = 'imu_link'
        
        # 【核心修正】：返璞归真，仅在 Z 轴暴力补回 9.80665 的重力，彻底解决“钻地”问题
        imu.linear_acceleration.x = float(row['accelx']) 
        imu.linear_acceleration.y = float(row['accely']) 
        imu.linear_acceleration.z = float(row['accelz']) + 9.80665 
 
        imu.angular_velocity.x = float(row['angvel_x'])
        imu.angular_velocity.y = float(row['angvel_y'])
        imu.angular_velocity.z = float(row['angvel_z'])
        
        writer.write(imu_topic, serialize_message(imu), t_sec * 10**9 + t_nano)

    print("\n🚀 开始多核加速打包 128 线 LiDAR 数据...")
    lidar_files = sorted(glob.glob(os.path.join(dataset_dir, 'lidar', '*.bin')))
    
    # 获取电脑的 CPU 核心数，保留 1 个核心给主进程写磁盘，避免死锁
    num_workers = max(1, cpu_count() - 1)
    print(f"🔥 正在启动 {num_workers} 个 CPU 核心进行并行点云处理...")

    # 包装目标函数，将 first_imu_time 固定进去
    process_func = partial(process_lidar_file, first_imu_time=first_imu_time)
    
    # 启用多进程池，imap 保证返回的序列化数据严格按照文件的时间顺序
    with Pool(processes=num_workers) as pool:
        for serialized_msg, timestamp in tqdm(pool.imap(process_func, lidar_files), total=len(lidar_files), desc="多核并发转换", unit="帧"):
            # 如果不是 None，说明该帧通过了“早产检测”，将其无脑写入磁盘
            if serialized_msg is not None:
                writer.write(lidar_topic, serialized_msg, timestamp)
                
    print(f"\n✅ ROS2 Bag 并行转换完成！输出路径: {output_bag}\n")

if __name__ == '__main__':
    # 【注意】：运行前请确认这里的输入路径和输出路径是你想要的
    input_dir = '/media/zzhe/T7/datasets/boreas-2021-07-20-17-33' 
    output_name = '/media/zzhe/T7/datasets/boreas-2021-07-20-17-33_bag'
    
    if os.path.exists(output_name):
        import shutil
        print(f"检测到已存在同名目录 {output_name}，正在清理...")
        shutil.rmtree(output_name)
        
    create_ros2_bag(input_dir, output_name)
