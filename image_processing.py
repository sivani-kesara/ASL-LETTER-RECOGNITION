import os
import cv2

minValue = 70
asl_dataset_path = 'data'
processed_images_path = 'processed_images'

if not os.path.exists(processed_images_path):
    os.makedirs(processed_images_path)

print("Starting image processing...")

for dataset_folder_name in os.listdir(asl_dataset_path):
    dataset_folder_path = os.path.join(asl_dataset_path, dataset_folder_name)
    processed_dataset_folder_path = os.path.join(processed_images_path, dataset_folder_name)

    if not os.path.exists(processed_dataset_folder_path):
        os.makedirs(processed_dataset_folder_path)

    for letter_folder_name in os.listdir(dataset_folder_path):
        letter_folder_path = os.path.join(dataset_folder_path, letter_folder_name)
        processed_letter_folder_path = os.path.join(processed_dataset_folder_path, letter_folder_name)

        if not os.path.exists(processed_letter_folder_path):
            os.makedirs(processed_letter_folder_path)

        for image_name in os.listdir(letter_folder_path):
            image_path = os.path.join(letter_folder_path, image_name)
            processed_image_path = os.path.join(processed_letter_folder_path, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            test_image = cv2.resize(test_image, (350, 350))

            cv2.imwrite(processed_image_path, test_image)

print("Finished processing all images.")
