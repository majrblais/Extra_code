#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np

def load_images_from_folder(folder, target_size=(512, 512)):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            images.append(resized_img)
            filenames.append(filename)
    return images, filenames

def main():
    # Prompt the user for input
    base_folder = input("Enter the path to the main folder: ").strip()
    subfolder1 = input("Enter the name of the first subfolder (e.g., 'images'): ").strip()
    subfolder2 = input("Enter the name of the second subfolder (e.g., 'images2'): ").strip()
    save_folder = input("Enter the path to the save folder (e.g., 'saved_images'): ").strip()

    # Print instructions
    print("\nInstructions:")
    print("1. The script will display images from both subfolders side by side, the masks (e.g images2) will be displayed in black/white but will be saved using the original format (0,1) so they will not be visible .")
    print("2. Use the 'a' key to navigate left and the 'd' key to navigate right.")
    print("3. Press 'y' to save the displayed pair of images.")
    print("4. Press 'n' to skip the displayed pair of images.")
    print("5. Press 'Esc' to exit the viewer.\n")

    # Create save folder if it does not exist
    os.makedirs(os.path.join(save_folder, subfolder1), exist_ok=True)
    os.makedirs(os.path.join(save_folder, subfolder2), exist_ok=True)

    # Define the target size for all images
    target_size = (512, 512)  # Width, Height in pixels

    # Load images from each subfolder into separate lists
    all_images = []
    all_titles = []
    for sub in [subfolder1, subfolder2]:
        images, filenames = load_images_from_folder(os.path.join(base_folder, sub), target_size)
        if sub == subfolder2:
            images = [img * 255 for img in images]  # Multiply images in the second subfolder by 255
        all_images.append(images)
        all_titles.append(filenames)

    # Find the maximum length to handle subfolders with different numbers of images
    max_length = max(len(images) for images in all_images)

    current_index = 0

    while True:
        displayed_images = []
        for idx, images in enumerate(all_images):
            if current_index < len(images):
                img = images[current_index]
                cv2.putText(img, all_titles[idx][current_index], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                displayed_images.append(img)
            else:
                displayed_images.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))

        vertical_separator = np.zeros((target_size[1], 10, 3), dtype=np.uint8)
        for i in range(1, len(displayed_images)):
            displayed_images.insert(2 * i - 1, vertical_separator)

        concatenated_image = np.hstack(displayed_images)

        cv2.imshow('Image Viewer', concatenated_image)

        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # Esc key to stop
            break
        elif k == ord('a'):  # 'a' key for left navigation
            if current_index > 0:
                current_index -= 1
        elif k == ord('d'):  # 'd' key for right navigation
            if current_index < max_length - 1:
                current_index += 1
        elif k == ord('y'):  # 'y' key to save images
            for idx, sub in enumerate([subfolder1, subfolder2]):
                source_file_path = os.path.join(base_folder, sub, all_titles[idx][current_index])
                target_file_path = os.path.join(save_folder, sub, all_titles[idx][current_index])
                if not os.path.exists(target_file_path):
                    os.rename(source_file_path, target_file_path)
            print(f"Saved: {all_titles[0][current_index]} and {all_titles[1][current_index]}")
        elif k == ord('n'):  # 'n' key to skip
            print(f"Skipped: {all_titles[0][current_index]} and {all_titles[1][current_index]}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[ ]:




