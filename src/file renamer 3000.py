import os

def rename_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return

    # Get all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Sort files to ensure consistent ordering
    files.sort()

    # Rename files sequentially
    for index, filename in enumerate(files):
        file_extension = os.path.splitext(filename)[1]  # Get the file extension
        new_name = f"{index:04d}{file_extension}"  # Generate name with 4 digits, e.g., 0000.png
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")

# Folder path (update this to your desired folder)
folder_path = "C:\\Users\\enjnir\\Desktop\\sorted_data\\9"

rename_files_in_folder(folder_path)
