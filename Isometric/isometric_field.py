

from math import sqrt, cos, sin, radians
from numpy import meshgrid, array, float_, matmul, linspace
from pygame import draw


class IsoField:
    def __init__(self, n, w):
        self.__n = n + 1
        x = linspace(-n/2, n/2, n + 1)
        z = linspace(-n/2, n/2, n + 1)
        self.__xyz = array(meshgrid(x, [0], z)).T.reshape(-1, 3).T

        sqrt6 = sqrt(6)
        sqrt3 = sqrt(3)
        sqrt2 = sqrt(2)

        self.__rot_mtx = array([
            [sqrt3,     0, -sqrt3],
            [1,         2,      1],
            [sqrt2, -sqrt2, sqrt2]],
            dtype=float_)
        self.__rot_mtx /= sqrt6
        self.__proj_mtx = array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]],
            dtype=float_)
        self.__shift_matrix = array([
            [w, 0, 640],
            [0, w, 320],
            [0, 0,   1]],
            dtype=float_)

        rad = radians(1)
        cosA = cos(rad)
        sinA = sin(rad)

        self.__r_mtx = array([
            [cosA, sinA, 0],
            [-sinA, cosA, 0],
            [0, 0, 1]],
            dtype=float_)

        self.__rotated_points = matmul(self.__rot_mtx, self.__xyz)
        projected_points = matmul(self.__proj_mtx, self.__rotated_points)
        projected_points[2, :] = 1
        self.__shifted_points = matmul(self.__shift_matrix, projected_points).T

    def update(self):
        self.__rot_mtx = matmul(self.__r_mtx, self.__rot_mtx)
        self.__rotated_points = matmul(self.__rot_mtx, self.__xyz)
        projected_points = matmul(self.__proj_mtx, self.__rotated_points)
        projected_points[2, :] = 1
        self.__shifted_points = matmul(self.__shift_matrix, projected_points).T

    def draw(self, screen):
        for idx in range(self.__shifted_points.shape[0] - 1):
            point = self.__shifted_points[idx]
            point_r = self.__shifted_points[idx + 1]

            if (idx + 1) % self.__n != 0:
                draw.line(screen, (255, 0, 0, 255), point[:-1], point_r[:-1])

            if idx + self.__n < self.__shifted_points.shape[0]:
                point_d = self.__shifted_points[idx + self.__n]
                draw.line(screen, (255, 0, 0, 255), point[:-1], point_d[:-1])


if __name__ == "__main__":
    ...
