# Path to the directories where you want to save the train and test images
train_dir_t2w = ""
train_dir_dwi = ""
train_dir_adc = ""
train_dir_clinical = ""

val_dir_t2w = ""
val_dir_dwi =  ""
val_dir_adc = ""
val_dir_clinical = ""

test_dir_t2w = ""
test_dir_dwi = ""
test_dir_adc = ""
test_dir_clinical = ""

def sort_files(file_list):
    return sorted(file_list, key=lambda x: os.path.basename(x).split('_')[0])




#training
# Load your CSV file with labels
label_train_df = pd.read_csv('')
# Create a dictionary mapping patient id to label
label_train_dict = pd.Series(label_train_df.label.values, index=label_train_df.patient_id).to_dict()

labels_train_cat = [label_train_dict[path.split('/')[-1].split('_')[0]] for path in train_list_t2w]
labels_train = [path.split('/')[-1].split('_')[0] for path in train_list_t2w]

#validation
# Load your CSV file with labels
label_val_df = pd.read_csv('')

# Create a dictionary mapping patient id to label
label_val_dict = pd.Series(label_val_df.label.values, index=label_val_df.patient_id).to_dict()

labels_val_cat = [label_val_dict[path.split('/')[-1].split('_')[0]] for path in val_list_t2w]
labels_val = [path.split('/')[-1].split('_')[0] for path in val_list_t2w]

#testing
label_test_df = pd.read_csv('')
label_test_dict = pd.Series(label_test_df.label.values, index=label_test_df.patient_id).to_dict()

labels_test_cat = [label_test_dict[path.split('/')[-1].split('_')[0]] for path in test_list_t2w]
labels_test = [path.split('/')[-1].split('_')[0] for path in test_list_t2w]

#augmentations
train_transforms = tio.Compose([
    tio.transforms.RandomFlip(flip_probability=0.4)

])
val_transforms = tio.Compose([
])
test_transforms = tio.Compose([
])

class Prostate_train_3D(Dataset):
    def __init__(self, t2w_list, dwi_list, adc_list, clinical_list, transform=None):
        self.t2w_list = t2w_list
        self.dwi_list = dwi_list
        self.adc_list = adc_list
        self.clinical_list = clinical_list
        self.transform = transform
        self.epoch_usage = set()  # Initialize the set to track image usage
        self.problematic_patients = []  # To track problematic patient IDs

    def __len__(self):
        return len(self.t2w_list)
    
    def normalize_sequence(self, img, patient_id):
        # Normalize the image to the range [0, 1], checking for problematic cases
        img_min = img.min()
        img_max = img.max()
        if img_max == img_min:
            # Record the patient ID if the image has constant values
            self.problematic_patients.append(patient_id)
            return img  # Return the original image or handle differently
        img_normalized = (img - img_min) / (img_max - img_min)
        return img_normalized

    def __getitem__(self, idx):
        t2w_path = self.t2w_list[idx]
        dwi_path = self.dwi_list[idx]
        adc_path = self.adc_list[idx]
        clinical_path = self.clinical_list[idx]

        t2w_img = np.load(t2w_path)
        t2w_img = t2w_img.reshape(1, 32, 224, 224)
        dwi_img = np.load(dwi_path)
        dwi_img = dwi_img.reshape(1, 32, 224, 224)
        adc_img = np.load(adc_path)
        adc_img = adc_img.reshape(1, 32, 224, 224)
        clinical_var = np.load(clinical_path)
        
        # Extract patient_id
        patient_id = t2w_path.split('/')[-1].split('_')[0]
        patient_id_dwi = dwi_path.split('/')[-1].split('_')[0]
        patient_id_adc = adc_path.split('/')[-1].split('_')[0]
        patient_id_clinical = clinical_path.split('/')[-1].split('_')[0]
        label = label_train_dict[patient_id]

        # Normalize each sequence independently
        t2w_img_normalized = self.normalize_sequence(t2w_img, patient_id)
        dwi_img_normalized = self.normalize_sequence(dwi_img, patient_id)
        adc_img_normalized = self.normalize_sequence(adc_img, patient_id)
        
        assert patient_id == patient_id_dwi == patient_id_adc == patient_id_clinical, "Patient_id's must be the same to stack the volumes"


        # Apply transforms
        if self.transform:
            t2w_img_norm_aug = self.transform(t2w_img_normalized)  # Apply transform
            dwi_img_norm_aug = self.transform(dwi_img_normalized)  # Apply transform
            adc_img_norm_aug = self.transform(adc_img_normalized)  # Apply transform


        return t2w_img_norm_aug, dwi_img_norm_aug, adc_img_norm_aug, clinical_var, label, patient_id

    def get_problematic_patients(self):
        return self.problematic_patients


class Prostate_val_3D(Dataset):

    def __init__(self, t2w_list, dwi_list, adc_list, clinical_list, transform=None):
        self.t2w_list = t2w_list
        self.dwi_list = dwi_list
        self.adc_list = adc_list
        self.clinical_list = clinical_list
        self.transform = transform
        self.epoch_usage = set()  # Initialize the set to track image usage
        self.problematic_patients = []  # To track problematic patient IDs
    
    def normalize_sequence(self, img, patient_id):
        # Normalize the image to the range [0, 1], checking for problematic cases
        img_min = img.min()
        img_max = img.max()
        if img_max == img_min:
            # Record the patient ID if the image has constant values
            self.problematic_patients.append(patient_id)
            return img  # Return the original image or handle differently
        img_normalized = (img - img_min) / (img_max - img_min)
        return img_normalized 

    def __len__(self):
        return len(self.t2w_list)

    def __getitem__(self, idx):
        t2w_path = self.t2w_list[idx]
        dwi_path = self.dwi_list[idx]
        adc_path = self.adc_list[idx]
        clinical_path = self.clinical_list[idx]

        t2w_img = np.load(t2w_path)
        t2w_img = t2w_img.reshape(1, 32, 224, 224)
        dwi_img = np.load(dwi_path)
        dwi_img = dwi_img.reshape(1, 32, 224, 224)
        adc_img = np.load(adc_path)
        adc_img = adc_img.reshape(1, 32, 224, 224)
        clinical_var = np.load(clinical_path)
        
        # Extract patient_id from the t2w_path
        patient_id = t2w_path.split('/')[-1].split('_')[0]
        patient_id_dwi = dwi_path.split('/')[-1].split('_')[0]
        patient_id_adc = adc_path.split('/')[-1].split('_')[0]
        patient_id_clinical = clinical_path.split('/')[-1].split('_')[0]
        
        # Normalize each sequence independently
        t2w_img_normalized = self.normalize_sequence(t2w_img, patient_id)
        dwi_img_normalized = self.normalize_sequence(dwi_img, patient_id)
        adc_img_normalized = self.normalize_sequence(adc_img, patient_id)
        
        assert patient_id == patient_id_dwi == patient_id_adc == patient_id_clinical, "Patient_id's must be the same to process the according t2w, dwi, adc"

            
        # Get label
        label = label_val_dict[patient_id]

        return t2w_img_normalized, dwi_img_normalized, adc_img_normalized, clinical_var, label, patient_id
    def get_problematic_patients(self):
        return self.problematic_patients



train_data = Prostate_train_3D(train_list_t2w, train_list_dwi, train_list_adc, train_list_clinical, transform=train_transforms)
train_loader = DataLoader(dataset = train_data, batch_size=2, shuffle=True, num_workers=4)

valid_data = Prostate_val_3D(val_list_t2w, val_list_dwi, val_list_adc, val_list_clinical, transform=None)
valid_loader = DataLoader(dataset = valid_data, batch_size=2, shuffle=False, num_workers=4)
