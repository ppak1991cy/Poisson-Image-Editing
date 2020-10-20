"""
    Poisson Image Editing
    func  : 完整代码
    Author: Chen Yu
    Date  : 2020.10.20
"""
import os
from enum import Enum

import numpy as np
from scipy.sparse import linalg as linalg
from scipy.sparse import lil_matrix as lil_matrix
import cv2


class PositionType(Enum):
    OMEGA = 0
    DELTA_OMEGA = 1
    OUTSIDE = 2


class PoissonImageEditing(object):

    def _get_neighbors(self, point_pos):
        row, col = point_pos
        return [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]

    def _judge_position(self, mask, point_pos):
        if mask[point_pos] != 1:
            return PositionType.OUTSIDE
        for neighbor_pos in self._get_neighbors(point_pos):
            if mask[neighbor_pos] != 1:
                return PositionType.DELTA_OMEGA
        return PositionType.OMEGA

    def _get_poisson_matrix(self, points_pos):
        point_num = len(points_pos)
        poisson_matrix = lil_matrix((point_num, point_num))
        for i, point_pos in enumerate(points_pos):
            poisson_matrix[i, i] = 4  # |Np|
            for neighbor_pos in self._get_neighbors(point_pos):
                if neighbor_pos in points_pos:
                    j = points_pos.index(neighbor_pos)
                    poisson_matrix[i, j] = -1
        return poisson_matrix

    def _get_guidance_filed(self, source, points_pos):
        guidance_filed = np.zeros(len(points_pos))
        for i, point_pos in enumerate(points_pos):
            row, col = point_pos
            sum_v = (4 * source[row, col]) - \
                source[row + 1, col] - \
                source[row - 1, col] - \
                source[row, col + 1] - \
                source[row, col - 1]
            guidance_filed[i] = sum_v
        return guidance_filed

    def _get_border_value(self, mask, target, points_pos):
        border_value = np.zeros(len(points_pos))
        for i, point_pos in enumerate(points_pos):
            if self._judge_position(mask, point_pos) == PositionType.DELTA_OMEGA:
                for neighbor_pos in self._get_neighbors(point_pos):
                    if self._judge_position(mask, neighbor_pos) == PositionType.OUTSIDE:
                        border_value[i] += target[neighbor_pos]
        return border_value

    def _process(self, source, target, mask):
        mask_pos = np.nonzero(mask)
        mask_pos = list(zip(mask_pos[0], mask_pos[1]))

        # 根据等式(7)，求解f
        poisson_matrix = self._get_poisson_matrix(mask_pos)
        guidance_filed = self._get_guidance_filed(source, mask_pos)
        border_value = self._get_border_value(mask, target, mask_pos)
        f = linalg.cg(poisson_matrix, guidance_filed + border_value)[0]
        new_target = np.copy(target).astype(np.int)
        for i, point_pos in enumerate(mask_pos):
            new_target[point_pos] = f[i]
        return new_target

    def synthesize(self, source_path, target_path, mask_path):
        source = cv2.imread(source_path, cv2.IMREAD_COLOR)
        target = cv2.imread(target_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float) / 255
        mask[mask != 1] = 0

        nchannel = source.shape[-1]
        channel_res = []
        for channel in range(nchannel):
            res = self._process(source[:, :, channel], target[:, :, channel], mask)
            channel_res.append(res)
        result = cv2.merge(channel_res)
        return result


if __name__ == '__main__':
    sample_dir = 'input/3'
    source_path = os.path.join(sample_dir, 'source.jpg')
    target_path = os.path.join(sample_dir, 'target.jpg')
    mask_path = os.path.join(sample_dir, 'mask.jpg')

    pie = PoissonImageEditing()
    result = pie.synthesize(source_path, target_path, mask_path)
    cv2.imwrite('result.png', result)
