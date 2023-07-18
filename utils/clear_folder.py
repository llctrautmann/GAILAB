import os
import shutil

def clear_folders(folder1, folder2, folder3):
    try:
        # Clear folder 1
        for filename in os.listdir(folder1):
            file_path = os.path.join(folder1, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        
        # Clear folder 2
        for filename in os.listdir(folder2):
            file_path = os.path.join(folder2, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        
        # Clear folder 3
        for filename in os.listdir(folder3):
            file_path = os.path.join(folder3, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        
        print("Content of the folders cleared successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

