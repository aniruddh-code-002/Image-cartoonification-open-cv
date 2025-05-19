import cv2
import numpy as np
import matplotlib.pyplot as plt

def cartoonify_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image.")
        return None

    # Resize the image for faster processing
    img = cv2.resize(img, (512, 512))

    # Apply bilateral filter to smooth the image while preserving edges
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert to grayscale and apply median blur
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)

    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray_blurred, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 9)

    # Step 4: Combine smoothed colors with detected edges
    cartoon = cv2.bitwise_and(smoothed, smoothed, mask=edges)

    return cartoon

def main():
    # Input image path
    image_path = input("Enter the path to your image file: ")

    # Process the image
    cartoon_image = cartoonify_image(image_path)

    if cartoon_image is not None:
        # Display the original and cartoonified images side by side
        original_image = cv2.imread(image_path)
        original_resized = cv2.resize(original_image, (512, 512))

        combined = np.hstack((original_resized, cartoon_image))

        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        plt.title("Original vs Cartoonified")
        plt.axis('off')
        plt.show()

        # Save the output if desired
        save_option = input("Do you want to save the cartoonified image? (y/n): ")
        if save_option.lower() == 'y':
            save_path = "cartoonified_image.jpg"
            cv2.imwrite(save_path, cartoon_image)
            print(f"Cartoonified image saved as {save_path}")

if __name__ == "__main__":
    main()
