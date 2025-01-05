import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json

r0_c = 4.5
z0_c = 0
phi0_c = 0
Ar_c = 1
Az_c = 0.5
wr_c = 1
wz_c = 1
wphi_c = 1
pr_c = 0
pz_c = 0

r0_n = 2
z0_n = 2
phi0_n = -3
Ar_n = 0.5
Az_n = 0.1
wr_n = 1
wz_n = 1
wphi_n = 1
pr_n = 0
pz_n = 0

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def cylindrical_to_cartesian(r, phi, z):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z


def camera_position(t):
    # Определите положение камеры
    r_c = r0_c + Ar_c * np.sin(wr_c * t + pr_c)
    phi_c = phi0_c + wphi_c * t
    z_c = z0_c + Az_c * np.sin(wz_c * t + pz_c)
    return cylindrical_to_cartesian(r_c, phi_c, z_c)


def camera_direction(t):
    # Определите направление камеры относительно ее позиции
    r_n = r0_n + Ar_n * np.sin(wr_n * t + pr_n)
    phi_n = phi0_n + wphi_n * t
    z_n = z0_n + Az_n * np.sin(wz_n * t + pz_n)
    return cylindrical_to_cartesian(r_n, phi_n, z_n)


def plot_3d_scene(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Платоновы тела
    for solid in data['PlatonicSolids']:
        vertices = np.array(solid['Vertices'], dtype=float)
        faces = solid['Faces']
        center = np.array(solid['Center'])
        radius = solid['CircumscribedSphereRadius']

        # Вычисляем текущий радиус, как максимальное расстояние от центральной точки до вершин
        current_radius = np.max(np.linalg.norm(vertices - np.mean(vertices, axis=0), axis=1))

        # Масштабируем вершины, чтобы они вписывались в новую сферу заданного радиуса
        scale_factor = radius / current_radius
        scaled_vertices = vertices * scale_factor

        # Смещаем вершины, чтобы центр тела был в нужной точке
        scaled_vertices += center

        # Отображение полигонов тела
        for face in faces:
            poly3d = [[scaled_vertices[vertice-1] for vertice in face]]
            ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        # Отображение центра и сферы
        ax.scatter(center[0], center[1], center[2], color='red', label='Center')

    # Пол
    floor_vertices = np.array(data['Floor']['Vertices'])
    floor_faces = data['Floor']['Faces']
    for face in floor_faces:
        floor_poly3d = [[floor_vertices[vertice] for vertice in face]]
        ax.add_collection3d(
            Poly3DCollection(floor_poly3d, facecolors='gray', linewidths=1, edgecolors='black', alpha=.5))

    # Траектория камеры
    t_values = np.linspace(0, 2 * np.pi, num=100)
    camera_positions = np.array([camera_position(t) for t in t_values])
    camera_directions = np.array([camera_direction(t) for t in t_values])

    ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], label='Camera Path', color='green')
    for pos, dir in zip(camera_positions, camera_directions):
        normalized_dir = dir / np.linalg.norm(dir)  # нормализуем вектор направления
        ax.quiver(pos[0], pos[1], pos[2], normalized_dir[0], normalized_dir[1], normalized_dir[2], length=0.5, color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.set_zlim([-6, 6])

    plt.legend()
    plt.show()


def main():
    file_path = 'coordinates.json'
    data = read_data(file_path)
    plot_3d_scene(data)


if __name__ == '__main__':
    main()
