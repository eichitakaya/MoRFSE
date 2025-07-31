import os
import glob
import pandas as pd
import cv2
import json
import numpy as np
import pydicom
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

def excluded_data():
    df = pd.read_csv("../data/csv/original_data.csv")
    df = df.query("classification == 'Exclusion' or classification == 'Invisible'")
    
    excluded_id_side = []
    
    for i in range(len(df)):
        num_lesion = df.iloc[i, 2]
        
        if pd.isnull(df.iloc[i, 13]):
            num_m_dist = 0
        else:
            num_m_dist = int(df.iloc[i, 13])
        
        if pd.isnull(df.iloc[i, 9]):
            num_b_dist = 0
        else:
            num_b_dist = int(df.iloc[i, 9])

        if pd.isnull(df.iloc[i, 13]) and pd.isnull(df.iloc[i, 9]):
            continue
        else:
            if num_lesion == num_m_dist + num_b_dist:
                excluded_id_side.append(df.iloc[i, 0])
            else:
                continue
    
    df_query = df[~df["id_side"].isin(excluded_id_side)]    
    df_query.to_csv("../data/csv/excluded_original_data.csv", index=False)
    
def get_annotation_center():
    id_side_list = []
    label_list = []
    x_center_list = []
    y_center_list = []

    annotation_files = glob.glob(os.path.join(root_dir, "*", "*", "*", "*", "AnnotationFile.Json"))
    dicom_files = glob.glob(os.path.join(root_dir, "*", "*", "*", "*", "DicomFile.dcm"))
    
    if len(annotation_files) != len(dicom_files):
        print("number of annotation files and dicom files are not equal.")

    root_dir = "../original_cmmmd/2-DecompressedZip"
    df = pd.read_csv("../data/csv/excluded_original_data.csv")

    # for malignant and benign
    df_query = df.query("classification != 'Normal'")
    
    for i in range(len(dicom_files)):
        dcm = pydicom.dcmread(dicom_files[i])
        
        patient_id = dcm[0x0010, 0x0020].value
        image_laterality = dcm[0x0020, 0x0062].value
        view = (dcm[0x0054, 0x0220].value)[0][0x0008, 0x0104].value
        
        if view == "medio-lateral oblique" and f"{patient_id}_{image_laterality}" in df_query["id_side"].values:
            with open(annotation_files[i]) as f:
                try:
                    d = json.load(f)
                    for annotation in d:
                        try:
                            annotation_points = annotation['cgPoints']
                            label = annotation['label']
                            
                            x_list = []
                            y_list = []
                            
                            for point in annotation_points:
                                x = point['x']
                                y = point['y']
                            
                                x_list.append(x)
                                y_list.append(y)
                            
                            x_center = sum(x_list) / len(x_list)
                            y_center = sum(y_list) / len(y_list)
                            
                            id_side_list.append(f"{patient_id}_{image_laterality}")
                            label_list.append(label)
                            x_center_list.append(x_center)
                            y_center_list.append(y_center)
                        
                        except IndexError:
                            print(f"IndexError: {patient_id}_{image_laterality}")
                    
                except json.decoder.JSONDecodeError:
                    print(f"JSONDecodeError: {patient_id}_{image_laterality}")
    
    # for normal
    df_query = df.query("classification == 'Normal'")

    for i in range(len(dicom_files)):
        dcm = pydicom.dcmread(dicom_files[i])
        
        patient_id = dcm[0x0010, 0x0020].value
        image_laterality = dcm[0x0020, 0x0062].value
        view = (dcm[0x0054, 0x0220].value)[0][0x0008, 0x0104].value
        
        if view == "medio-lateral oblique" and f"{patient_id}_{image_laterality}" in df_query["id_side"].values:
            img = dcm.pixel_array
            
            pixels_ = np.copy(img)
            amin = np.amin(pixels_)
            amax = np.amax(pixels_)
            pixelsByte = ((pixels_-amin)/(amax-amin)) * 255
            pixelsByte = np.clip(pixelsByte, 0, 255)
            pixelsByte = np.uint8(pixelsByte)

            _, binary = cv2.threshold(pixelsByte, 1, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contour = contours[0]
            
            while True:
                x_center = np.random.randint(contour[:, 0, 0].min(), contour[:, 0, 0].max())
                y_center = np.random.randint(contour[:, 0, 1].min(), contour[:, 0, 1].max())
                
                if cv2.pointPolygonTest(contour, (x_center, y_center), False) >= 0:
                    x_start = max(0, x_center - 256)
                    x_end = x_center + 256
                    y_start = y_center - 256
                    y_end = y_center + 256
                    
                    if x_start < 0 or x_end > 2294 or y_start < 0 or y_end > 2294:
                        continue
                    
                    cropped_img = pixelsByte[y_start:y_end, x_start:x_end]

                    id_side_list.append(f"{patient_id}_{image_laterality}")
                    label_list.append("Normal")
                    x_center_list.append(x_center)
                    y_center_list.append(y_center)
                    break 

    output_df = pd.DataFrame({
        "id_side": id_side_list,
        "label": label_list,
        "x_center": x_center_list,
        "y_center": y_center_list
    })
    
    output_df.to_csv("../data/csv/annotation_center.csv", index=False)

def data_crop(annotation_center_csv_path, cmmd_csv_path, root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    center_df = pd.read_csv(annotation_center_csv_path)
    cmmd_df = pd.read_csv(cmmd_csv_path)
    
    img_paths = glob.glob(os.path.join(root_dir, "*.png"))
    
    for path in img_paths:
        id_side = os.path.basename(path)[:9]
        lesion_category = os.path.basename(path).split("_")[2]
        
        if lesion_category == "Normal":
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contour = contours[0]
            
            x_center = int(contour[:, 0, 0].mean())
            y_center = int(contour[:, 0, 1].mean())
            
            x_start = x_center - 256
            x_end = x_center + 256
            y_start = y_center - 256
            y_end = y_center + 256
            
            cropped_img = img[y_start:y_end, x_start:x_end]
            cv2.imwrite(f"{output_dir}/{id_side}_Normal_normal_1.png", cropped_img)

        elif lesion_category == "Malignant":
            lesion_category = "Malign"
        
        center_df_query = center_df.query(f"id_side == '{id_side}'")
        cmmd_df_query = cmmd_df.query(f"id_side == '{id_side}'")
        
        num_lesion = cmmd_df_query.iloc[0, 2]
        
        if num_lesion != len(center_df_query):
            print(f"number of lesion is not equal to the number of annotation: {id_side}")
            continue
        
        lesion_types = []
        
        idx_to_lesion_type = {
            4: "b-FA--",
            5: "b-lipo",
            6: "b-mass",
            7: "b-calc",
            8: "b-FAD-",
            9: "b-dist",
            10: "m-mass",
            11: "m-calc",
            12: "m-FAD-",
            13: "m-dist"
        }
        
        for i in range(4, 14):
            if not pd.isnull(cmmd_df_query.iloc[0, i]):
                for j in range(int(cmmd_df_query.iloc[0, i])):
                    lesion_types.append(idx_to_lesion_type[i])
        
        for i in range(len(center_df_query)):
            x_center = center_df_query.iloc[i, 4]
            y_center = center_df_query.iloc[i, 3]
            annotation_label = center_df_query.iloc[i, 1]
            
            x1 = int(x_center - 256)
            y1 = int(y_center - 256)
            x2 = int(x_center + 256)
            y2 = int(y_center + 256)
            
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Image not found or unable to read: {path}")
                continue

            height, width = img.shape
            cropped_img = np.zeros((512, 512), dtype=np.uint8)

            x_start = max(0, -x1)
            y_start = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            cropped_img[y_start:y_start + (y2 - y1), x_start:x_start + (x2 - x1)] = img[y1:y2, x1:x2]

            lesion_label = None
            if annotation_label == "mass":
                if "m-mass" in lesion_types:
                    lesion_label = "m-mass"
                    lesion_types.remove("m-mass")
                elif "b-mass" in lesion_types:
                    lesion_label = "b-mass"
                    lesion_types.remove("b-mass")
            elif annotation_label == "calc":
                if "m-calc" in lesion_types:
                    lesion_label = "m-calc"
                    lesion_types.remove("m-calc")
                elif "b-calc" in lesion_types:
                    lesion_label = "b-calc"
                    lesion_types.remove("b-calc")
            elif annotation_label == "FAD":
                if "m-FAD-" in lesion_types:
                    lesion_label = "m-FAD-"
                    lesion_types.remove("m-FAD-")
                elif "b-FAD-" in lesion_types:
                    lesion_label = "b-FAD-"
                    lesion_types.remove("b-FAD-")
            elif annotation_label == "FA":
                if "b-FA--" in lesion_types:
                    lesion_label = "b-FA--"
                    lesion_types.remove("b-FA--")
            elif annotation_label == "lipoma":
                if "b-lipo" in lesion_types:
                    lesion_label = "b-lipo"
                    lesion_types.remove("b-lipo")
            elif annotation_label == "dist":
                if "m-dist" in lesion_types:
                    lesion_label = "m-dist"
                    lesion_types.remove("m-dist")
                elif "b-dist" in lesion_types:
                    lesion_label = "b-dist"
                    lesion_types.remove("b-dist")

            output_path = os.path.join(output_dir, f"{id_side}_{lesion_category}_{lesion_label}_{num_lesion}-{i+1}.png")
            cv2.imwrite(output_path, cropped_img)     

def data_divide(root_dir, output_dir, n_splits=5):
    img_paths = glob.glob(os.path.join(root_dir, "*.png"))
    
    img_paths = [img_path for img_path in img_paths if "m-dist" not in img_path and "b-dist" not in img_path]
    
    type_to_category = {
        "m-mass": 0,
        "b-mass": 1,
        "b-FA--": 1,
        "b-lipo": 1,
        "m-calc": 0,
        "b-calc": 1,
        "m-FAD-": 0,
        "b-FAD-": 2,
        "normal": 2
    }
    
    type_to_type = {
        "m-mass": 0,
        "b-mass": 0,
        "b-FA--": 0,
        "b-lipo": 0,
        "m-calc": 1,
        "b-calc": 1,
        "m-FAD-": 0,
        "b-FAD-": 2,
        "normal": 2
    }
    
    df = pd.DataFrame(img_paths, columns=["img_path"])
    df["patient_id"] = df["img_path"].apply(lambda x: os.path.basename(x)[:7])
    df["lesion_category"] = df["img_path"].apply(lambda x: type_to_category[x.split("/")[-1].split("_")[3]])
    df["lesion_type"] = df["img_path"].apply(lambda x: type_to_type[x.split("/")[-1].split("_")[3]])
    
    patient_groups = df.groupby("patient_id")["lesion_category"].apply(lambda x: x.mode()[0]).reset_index()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    patient_groups["fold"] = -1

    for fold_number, (_, test_index) in enumerate(skf.split(patient_groups["patient_id"], patient_groups["lesion_category"])):
        patient_groups.loc[test_index, "fold"] = fold_number

    df = df.merge(patient_groups[["patient_id", "fold"]], on="patient_id", how="left")
    df.to_csv(os.path.join(output_dir, "data_divide.csv"), index=False)
    
def check_divide(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    for fold_number in range(df["fold"].nunique()):
        print(f"fold_{fold_number}")
        print(df[df["fold"] == fold_number]["lesion_category"].value_counts())
        print("\n")  
    
    fold_patient_ids = defaultdict(set)
    
    for fold in df["fold"].unique():
        fold_patient_ids[fold] = set(df[df["fold"] == fold]["patient_id"])
    
    duplicates = defaultdict(list)
    folds = list(fold_patient_ids.keys())
    
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            common_ids = fold_patient_ids[folds[i]].intersection(fold_patient_ids[folds[j]])
            if common_ids:
                duplicates[(folds[i], folds[j])] = common_ids
    
    if duplicates:
        for (fold1, fold2), patient_ids in duplicates.items():
            print(f"Duplicate patient IDs between fold {fold1} and fold {fold2}: {", ".join(patient_ids)}")
    else:
        print("All folds have unique patient IDs.")

if __name__ == "__main__":
    excluded_data()   
    get_annotation_center()

    annotation_center_csv = "../data/csv/annotation_center.csv"
    cmmd_csv = "../data/csv/excluded_original_data.csv"
    root_dir = "../data/original_img"
    output_dir = "../data/cropped_img"
    data_crop(annotation_center_csv, cmmd_csv, root_dir, output_dir)

    root_dir = "../data/cropped_img"
    output_dir = "../data/csv"
    data_divide(root_dir, output_dir, n_splits=5)
    check_divide(os.path.join(output_dir, "data_divide.csv"))
    
    