from PIL import Image
import os


def downscale_images(folder_path, output_folder, size=(28, 28)):
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it is a valid image file
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    # Downscale the image
                    img_resized = img.resize(size, Image.Resampling.NEAREST)

                    # Save to the output folder with the same name
                    output_path = os.path.join(output_folder, filename)
                    img_resized.save(output_path)
                    print(f"Downscaled and saved: {output_path}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")


# Folder paths (update these as needed)
input_folder = "C://Users//enjnir//Desktop//sorted_data//notdownscaled_9"
output_folder = "C://Users//enjnir//Desktop//sorted_data//9"

# Downscale images
downscale_images(input_folder, output_folder)
