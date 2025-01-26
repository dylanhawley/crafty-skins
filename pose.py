import cv2
import mediapipe as mp
import numpy as np
import os

def get_body_part_image(landmark_indices, image, landmarks, w, h, padding=10):
    """
    Crops the image around the specified body part landmarks.

    Args:
        landmark_indices (list): Indices of the landmarks for the body part.
        image (ndarray): The original image.
        landmarks (list): List of landmarks detected by mediapipe.
        w (int): Width of the image.
        h (int): Height of the image.
        padding (int): Padding to add around the cropped area.

    Returns:
        ndarray: Cropped image of the body part.
    """
    x_coords = [int(landmarks[i].x * w) for i in landmark_indices]
    y_coords = [int(landmarks[i].y * h) for i in landmark_indices]
    x_min = max(0, min(x_coords) - padding)
    x_max = min(w, max(x_coords) + padding)
    y_min = max(0, min(y_coords) - padding)
    y_max = min(h, max(y_coords) + padding)
    return image[y_min:y_max, x_min:x_max]

def main():
    image_path = 'person2.jpg'
    outdir = image_path.replace('.jpg', '') + '_body_parts'
    os.makedirs(outdir, exist_ok=True)
    # Load image
    image = cv2.imread(image_path)  # Replace with your image path

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Check if any landmarks are detected
        if not results.pose_landmarks:
            print("No pose detected in the image.")
            return

        landmarks = results.pose_landmarks.landmark
        h, w, c = image.shape

        # Define landmark indices for each body part
        head_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        torso_indices = [11, 12, 23, 24]
        left_arm_indices = [11, 13, 15, 17, 19, 21]
        right_arm_indices = [12, 14, 16, 18, 20, 22]
        left_leg_indices = [23, 25, 27, 29, 31]
        right_leg_indices = [24, 26, 28, 30, 32]

        # Get images of body parts
        head_image = get_body_part_image(head_indices, image, landmarks, w, h)
        cv2.imwrite(os.path.join(outdir, 'head_image.jpg'), head_image)

        torso_image = get_body_part_image(torso_indices, image, landmarks, w, h, padding=20)
        cv2.imwrite(os.path.join(outdir, 'torso_image.jpg'), torso_image)

        left_arm_image = get_body_part_image(left_arm_indices, image, landmarks, w, h)
        cv2.imwrite(os.path.join(outdir, 'left_arm_image.jpg'), left_arm_image)

        right_arm_image = get_body_part_image(right_arm_indices, image, landmarks, w, h)
        cv2.imwrite(os.path.join(outdir, 'right_arm_image.jpg'), right_arm_image)

        left_leg_image = get_body_part_image(left_leg_indices, image, landmarks, w, h)
        cv2.imwrite(os.path.join(outdir, 'left_leg_image.jpg'), left_leg_image)

        right_leg_image = get_body_part_image(right_leg_indices, image, landmarks, w, h)
        cv2.imwrite(os.path.join(outdir, 'right_leg_image.jpg'), right_leg_image)

        print("Cropped body parts have been saved successfully.")

if __name__ == '__main__':
    main()
