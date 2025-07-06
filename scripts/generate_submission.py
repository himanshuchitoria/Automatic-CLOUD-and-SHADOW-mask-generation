# scripts/generate_submission.py

import os
import zipfile

def create_submission_zip(output_dir, submission_zip_path):
    """
    Create a zip file for the final submission package.

    Args:
        output_dir (str): Directory containing all submission files and folders.
        submission_zip_path (str): Path to save the final zip file.
    """
    # Ensure the directory for the zip file exists
    os.makedirs(os.path.dirname(submission_zip_path), exist_ok=True)

    with zipfile.ZipFile(submission_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=output_dir)
                zipf.write(file_path, arcname)
    print(f"Created submission zip file at: {submission_zip_path}")

if __name__ == "__main__":
    # Directory containing all required submission files and folders
    output_dir = 'submission/'  # Must follow the NRSC structure:
                                # submission/
                                # ├── Report.pdf
                                # ├── Training.csv
                                # ├── Requirements.txt
                                # ├── Training_Labeled_data.zip
                                # ├── Inference_Code.zip
                                # ├── Model.zip
                                # └── [DatasetId]/
                                #     ├── mask.tiff
                                #     ├── cloudshapes.zip
                                #     └── shadowshapes.zip

    submission_zip_path = 'submission/Output.zip'
    create_submission_zip(output_dir, submission_zip_path)
