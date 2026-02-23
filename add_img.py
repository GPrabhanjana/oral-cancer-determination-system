import os
import shutil

# ==========================
# HARD CODE YOUR PATHS HERE
# ==========================

SOURCE_FOLDER = r"CBIR module\Oral Images Dataset\original_data\NON CANCER"
DESTINATION_FOLDER = r"CBIR module\Oral Images Dataset\original_data\benign_lesions"


def get_max_number(folder_path):
    max_num = 0
    for filename in os.listdir(folder_path):
        name, ext = os.path.splitext(filename)
        if name.isdigit():
            max_num = max(max_num, int(name))
    return max_num


def move_and_rename(source_folder, destination_folder):
    # Ensure destination exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get highest numbered image in destination
    current_max = get_max_number(destination_folder)

    # Collect numeric image files from source
    files = []
    for filename in os.listdir(source_folder):
        name, ext = os.path.splitext(filename)
        if name.isdigit() and ext.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            files.append((int(name), filename))

    # Sort numerically
    files.sort()

    # Copy and rename sequentially
    for number, filename in files:
        current_max += 1
        ext = os.path.splitext(filename)[1]
        new_name = f"{current_max}{ext}"

        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, new_name)

        shutil.copy(src_path, dst_path)

    print("Done. Images copied and renamed successfully.")


if __name__ == "__main__":
    move_and_rename(SOURCE_FOLDER, DESTINATION_FOLDER)