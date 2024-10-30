import pdb
import glob
import cv2
import os
import numpy as np

class PanoramaBuilder:
    def __init__(self):
        self.feature_extractor = cv2.SIFT_create()
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_L2)

    def create_panorama_from_directory(self, directory):
        image_files = glob.glob(f'{directory}/*.*')
        if len(image_files) < 2:
            raise ValueError("A minimum of two images is required to build a panorama")

        images = [cv2.imread(img) for img in image_files]
        panorama_image = images[0]
        homography_list = []

        for i in range(1, len(images)):
            kp1, des1 = self.feature_extractor.detectAndCompute(panorama_image, None)
            kp2, des2 = self.feature_extractor.detectAndCompute(images[i], None)

            knn_matches = self.feature_matcher.knnMatch(des1, des2, k=2)

            valid_matches = []
            for m, n in knn_matches:
                if m.distance < 0.75 * n.distance:
                    valid_matches.append(m)

            points_img1 = np.float32([kp1[m.queryIdx].pt for m in valid_matches]).reshape(-1, 2)
            points_img2 = np.float32([kp2[m.trainIdx].pt for m in valid_matches]).reshape(-1, 2)

            H = self.find_homography(points_img1, points_img2)
            if H is not None:
                homography_list.append(H)
                panorama_image = self.apply_inverse_warp(panorama_image, images[i], H)

        return panorama_image, homography_list

    def normalize_coordinates(self, points):
        mean = np.mean(points, axis=0)
        stdev = np.std(points, axis=0)
        stdev[stdev < 1e-8] = 1e-8  
        scale = np.sqrt(2) / stdev
        transform_matrix = np.array([[scale[0], 0, -scale[0] * mean[0]],
                                     [0, scale[1], -scale[1] * mean[1]],
                                     [0, 0, 1]])
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        normalized_points = (transform_matrix @ homogeneous_points.T).T
        return normalized_points[:, :2], transform_matrix

    def direct_linear_transform(self, src_points, dst_points):
        normalized_src, T1 = self.normalize_coordinates(src_points)
        normalized_dst, T2 = self.normalize_coordinates(dst_points)
        A = []
        for i in range(len(normalized_src)):
            x, y = normalized_src[i]
            x_prime, y_prime = normalized_dst[i]
            A.extend([[-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime],
                      [0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime]])
        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        H_normalized = Vt[-1].reshape(3, 3)
        homography = np.linalg.inv(T2) @ H_normalized @ T1      
        return homography / homography[2, 2]

    def find_homography(self, src_points, dst_points):
        max_iterations, inlier_threshold = 2000, 3.0
        best_H, max_inliers = None, 0

        if len(src_points) < 4:
            return None

        for _ in range(max_iterations):
            sample_indices = np.random.choice(len(src_points), 4, replace=False)
            sampled_src = src_points[sample_indices]
            sampled_dst = dst_points[sample_indices]

            candidate_H = self.direct_linear_transform(sampled_src, sampled_dst)
            if candidate_H is None:
                continue

            src_pts_homogeneous = np.hstack((src_points, np.ones((src_points.shape[0], 1))))
            projected_dst = (candidate_H @ src_pts_homogeneous.T).T

            projected_dst[projected_dst[:, 2] == 0, 2] = 1e-10
            projected_pts = projected_dst[:, :2] / projected_dst[:, 2, np.newaxis]

            residuals = np.linalg.norm(dst_points - projected_pts, axis=1)
            inliers = np.where(residuals < inlier_threshold)[0]

            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                best_H = candidate_H

        if best_H is not None and max_inliers >= 10:
            best_H = self.direct_linear_transform(src_points[inliers], dst_points[inliers])
        return best_H

    def apply_inverse_warp(self, img1, img2, H):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        transformed_corners = self.apply_homography_to_points(H, corners_img2)
        all_corners = np.vstack((transformed_corners, [[0, 0], [w1, 0], [w1, h1], [0, h1]]))
        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        H_translated = translation @ H
        output_size = (y_max - y_min, x_max - x_min)
        warped_image = self.warp_image(img1, img2, H_translated, output_size)

        stitched_img = np.zeros((output_size[0], output_size[1], 3), dtype=img1.dtype)
        stitched_img[-y_min:h1-y_min, -x_min:w1-x_min] = img1

        mask1 = (stitched_img > 0).astype(np.float32)
        mask2 = (warped_image > 0).astype(np.float32)
        final_mask = mask1 + mask2
        stitched_img = (stitched_img * mask1 + warped_image * mask2) / final_mask
        stitched_img = np.nan_to_num(stitched_img).astype(np.uint8)

        return stitched_img
