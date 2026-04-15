import open3d as o3d
import numpy as np

def visualize_kitti_bin(bin_path):
    # 1. .bin 파일 읽기 (KITTI는 float32 4개씩 묶여 있음)
    # x, y, z, intensity 순서
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    
    # 2. Open3D용 포인트클라우드 객체 생성
    pcd = o3d.geometry.PointCloud()
    
    # 3. 좌표 데이터(x, y, z)만 넣기
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    
    # 4. 반사도(Intensity)를 색상으로 표현 (선택사항)
    # 반사도가 높을수록 밝게 보이게 설정
    intensities = point_cloud[:, 3]
    colors = np.zeros((len(intensities), 3))
    colors[:, 0] = intensities / np.max(intensities) # Red 채널에 할당
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 5. 시각화 실행
    print(f"Visualizing: {bin_path}")
    o3d.visualization.draw_geometries([pcd])

# 여기에 보고 싶은 bin 파일 경로를 넣으세요
visualize_kitti_bin("data/kitti/training/velodyne/000000.bin")