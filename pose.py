import cv2
import mediapipe as mp
import numpy as np
from skinpy import Skin
from PIL import Image
import io

def get_body_part_image(landmark_indices, image, landmarks, w, h, padding=10):
    """
    Crops the image around the specified body part landmarks and returns as PIL Image.
    """
    x_coords = [int(landmarks[i].x * w) for i in landmark_indices]
    y_coords = [int(landmarks[i].y * h) for i in landmark_indices]
    x_min = max(0, min(x_coords) - padding)
    x_max = min(w, max(x_coords) + padding)
    y_min = max(0, min(y_coords) - padding)
    y_max = min(h, max(y_coords) + padding)
    
    cropped = image[y_min:y_max, x_min:x_max]
    # Convert BGR to RGB and then to PIL Image
    rgb_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_cropped)

def resize_to_skin_part(image, target_size):
    """Resize image to target size while maintaining aspect ratio and centering"""
    target_w, target_h = target_size
    aspect = image.size[0] / image.size[1]
    
    if aspect > target_w / target_h:  # wider than tall
        new_w = target_w
        new_h = int(target_w / aspect)
    else:  # taller than wide
        new_h = target_h
        new_w = int(target_h * aspect)
    
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create new image with target size
    final = Image.new('RGBA', target_size, (0, 0, 0, 0))
    # Paste resized image in center
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    final.paste(resized, (x, y))
    
    return final

def main():
    image_path = 'person2.jpg'
    image = cv2.imread(image_path)

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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

        # Get body part images as PIL Images
        head = get_body_part_image(head_indices, image, landmarks, w, h)
        torso = get_body_part_image(torso_indices, image, landmarks, w, h, padding=20)
        left_arm = get_body_part_image(left_arm_indices, image, landmarks, w, h)
        right_arm = get_body_part_image(right_arm_indices, image, landmarks, w, h)
        left_leg = get_body_part_image(left_leg_indices, image, landmarks, w, h)
        right_leg = get_body_part_image(right_leg_indices, image, landmarks, w, h)

        # Create a new skin
        skin = Skin.from_path("steve.png")  # Use steve.png as template
        
        # Resize and apply each body part to the appropriate skin section
        # Head (8x8)
        skin.head.front = resize_to_skin_part(head, (8, 8))
        skin.head.right = resize_to_skin_part(head, (8, 8))
        skin.head.left = resize_to_skin_part(head, (8, 8))
        skin.head.top = resize_to_skin_part(head, (8, 8))
        
        # Body (8x12)
        skin.body.front = resize_to_skin_part(torso, (8, 12))
        skin.body.back = resize_to_skin_part(torso, (8, 12))
        
        # Arms (4x12)
        skin.right_arm.front = resize_to_skin_part(right_arm, (4, 12))
        skin.right_arm.back = resize_to_skin_part(right_arm, (4, 12))
        skin.left_arm.front = resize_to_skin_part(left_arm, (4, 12))
        skin.left_arm.back = resize_to_skin_part(left_arm, (4, 12))
        
        # Legs (4x12)
        skin.right_leg.front = resize_to_skin_part(right_leg, (4, 12))
        skin.right_leg.back = resize_to_skin_part(right_leg, (4, 12))
        skin.left_leg.front = resize_to_skin_part(left_leg, (4, 12))
        skin.left_leg.back = resize_to_skin_part(left_leg, (4, 12))

        # Save the skin
        skin.save("output_skin.png")
        print("Minecraft skin has been created successfully.")

if __name__ == '__main__':
    main()
