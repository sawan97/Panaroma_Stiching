import pdb
import glob
import cv2
import os
import numpy as np

class PanaromaStitcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def make_panaroma_for_images_in(self, path):
        img_path = glob.glob('{}/*.*'.format(path))
        if len(img_path) < 2:
            raise ValueError("Need at least two images to create a panorama")

        images = [cv2.imread(im_path) for im_path in img_path]
        # if any(image is None for image in images):
        #     raise ValueError("Error reading one or more images from the path")

        stiched_image = images[0]
        homography_matrices = []

        for i in range(1, len(images)):
            kp1, des1 = self.sift.detectAndCompute(stiched_image, None)
            kp2, des2 = self.sift.detectAndCompute(images[i], None)

            knn_matches = self.matcher.knnMatch(des1, des2, k=2)

            good_matches = []
            for m, n in knn_matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # if len(good_matches) < 6:
            #     print(f"Warning: Not enough good matches between image {i} and image {i-1}. Skipping this pair.")
            #     continue

            points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            H = self.compute_homography(points1, points2)
            # if H is None:
            #     print(f"Warning: Failed to compute homography for image {i} and image {i-1}. Skipping this pair.")
            #     continue

            homography_matrices.append(H)
            stiched_image = self.inverse_warp(stiched_image, images[i], H)

        return stiched_image, homography_matrices

    def normalize_points(self, points):
        mean = np.mean(points, axis=0)
        stdDev = np.std(points, axis=0)
        stdDev[stdDev < 1e-8] = 1e-8 
        scale = np.sqrt(2) / stdDev
        T = np.array([[scale[0], 0, -scale[0]*mean[0]],
                      [0, scale[1], -scale[1]*mean[1]],
                      [0, 0, 1]])
        homogen_pts = np.hstack((points, np.ones((points.shape[0], 1))))
        normalized_pts = (T @ homogen_pts.T).T
        return normalized_pts[:, :2], T

    def dlt(self, points1, points2):
        points1_norm, T1 = self.normalize_points(points1)
        points2_norm, T2 = self.normalize_points(points1)
        A = []
        for i in range(len(points1_norm)):
            x, y = points1_norm[i]
            x_prime, y_prime = points2_norm[i]
            A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
            A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
        A = np.array(A)
        try:
            U, S, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            print("Warning: SVD did not converge. Returning None for homography.")
            return None
        H_norm = Vt[-1].reshape(3, 3)
        H = np.linalg.inv(T2) @ H_norm @ T1      
        return H / H[2, 2]

    def compute_homography(self, points1, points2):
        max_iterations = 2000
        threshold = 3.0 
        best_H = None
        max_inliers = 0
        best_inliers = []

        if len(points1) < 4:
            return None

        for _ in range(max_iterations):
            idx = np.random.choice(len(points1), 4, replace=False)
            p1_sample = points1[idx]
            p2_sample = points2[idx]

            H_candidate = self.dlt(p1_sample, p2_sample)
            if H_candidate is None:
                continue

            pts1_homogeneous = np.hstack((points1, np.ones((points1.shape[0], 1))))
            projected_pts2_homogeneous = (H_candidate @ pts1_homogeneous.T).T

            projected_pts2_homogeneous[projected_pts2_homogeneous[:, 2] == 0, 2] = 1e-10
            projected_pts2 = projected_pts2_homogeneous[:, :2] / projected_pts2_homogeneous[:, 2, np.newaxis]

            errors = np.linalg.norm(points2 - projected_pts2, axis=1)
            inliers = np.where(errors < threshold)[0]

            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                best_H = H_candidate
                best_inliers = inliers

        if best_H is not None and len(best_inliers) >= 10:  #atleast 10 inliers
            best_H = self.dlt(points1[best_inliers], points2[best_inliers])
        # else:
        #     print("Warning: Not enough inliers after RANSAC.")
        #     return None

        return best_H

    def apply_homography_to_points(self, H, pts):
        homogen_pts = np.hstack([pts, np.ones((pts.shape[0], 1))])
        transformed_pts = (H @ homogen_pts.T).T
        transformed_pts[transformed_pts[:, 2] == 0, 2] = 1e-10
        transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2, np.newaxis]
        return transformed_pts

    def warp_image(self, img1, img2, H, output_shape):
        h_out, w_out = output_shape   
        xx, yy = np.meshgrid(np.arange(w_out), np.arange(h_out))
        ones = np.ones_like(xx)
        coords = np.stack([xx, yy, ones], axis=-1).reshape(-1, 3)

        H_inv = np.linalg.inv(H)
        coords_transformed = coords @ H_inv.T
        coords_transformed[coords_transformed[:, 2] == 0, 2] = 1e-10
        coords_transformed /= coords_transformed[:, 2, np.newaxis]

        x_src = coords_transformed[:, 0]  
        y_src = coords_transformed[:, 1]

        valid_indices = (
            (x_src >= 0) & (x_src < img2.shape[1] - 1) &
            (y_src >= 0) & (y_src < img2.shape[0] - 1)
        )

        x_src = x_src[valid_indices]
        y_src = y_src[valid_indices]
        x0 = np.floor(x_src).astype(np.int32)
        y0 = np.floor(y_src).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        wx = x_src - x0        
        wy = y_src - y0

        img_flat = img2.reshape(-1, img2.shape[2])
        indices = y0 * img2.shape[1] + x0
        Ia = img_flat[indices]
        Ib = img_flat[y0 * img2.shape[1] + x1]
        Ic = img_flat[y1 * img2.shape[1] + x0]
        Id = img_flat[y1 * img2.shape[1] + x1]

        wa = (1 - wx) * (1 - wy)
        wb = wx * (1 - wy)
        wc = (1 - wx) * wy
        wd = wx * wy
        warped_pixels = (Ia * wa[:, np.newaxis] + Ib * wb[:, np.newaxis] +
                         Ic * wc[:, np.newaxis] + Id * wd[:, np.newaxis])

        # output image
        warped_image = np.zeros((h_out * w_out, img2.shape[2]), dtype=img2.dtype)
        warped_image[valid_indices] = warped_pixels
        warped_image = warped_image.reshape(h_out, w_out, img2.shape[2])

        return warped_image

    def inverse_warp(self, img1, img2, H):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        transformed_corners = self.apply_homography_to_points(H, corners_img2)
        all_corners = np.vstack((transformed_corners, [[0, 0], [w1, 0], [w1, h1], [0, h1]]))
        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

        translation = np.array([[1, 0, -x_min],
                                [0, 1, -y_min],
                                [0, 0, 1]])

        H_translated = translation @ H

        output_shape = (y_max - y_min, x_max - x_min)
        warped_img2 = self.warp_image(img1, img2, H_translated, output_shape)
        stiched_image = np.zeros((output_shape[0], output_shape[1], 3), dtype=img1.dtype)
        stiched_image[-y_min:-y_min + h1, -x_min:-x_min + w1] = img1

        # masks
        mask1 = (stiched_image > 0).astype(np.float32)
        mask2 = (warped_img2 > 0).astype(np.float32)

        # Blend images
        total_mask = mask1 + mask2
        stiched_image = (stiched_image * mask1 + warped_img2 * mask2) / total_mask
        stiched_image = np.nan_to_num(stiched_image).astype(np.uint8)

        return stiched_image
