from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import os
import random
import numpy as np
import glob
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import torch.nn.functional as F
import io
import nibabel
import shutil
import pickle
import torchio as tio
import math
from functools import partial

class Model(nn.Module):
    def __init__(self, in_channel, label_category_dict, dim=3):
        super(Model, self).__init__()
        # Create separate encoders for each input sequnece
        self.encoder_t2w = VGGEncoder(in_channel, [64, 128, 256, 512], [1, 2, 3, 4], [0.5, 0.5, 0.5, 0.5], norm='bn', active='relu', dim=dim)
        self.encoder_dwi = VGGEncoder(in_channel, [64, 128, 256, 512], [1, 2, 3, 4], [0.5, 0.5, 0.5, 0.5], norm='bn', active='relu', dim=dim)
        self.encoder_adc = VGGEncoder(in_channel, [64, 128, 256, 512], [1, 2, 3, 4], [0.5, 0.5, 0.5, 0.5], norm='bn', active='relu', dim=dim)
 
        self.cross_attention = CrossAttention(feature_dim=512, num_heads=4, dropout=0.2)
        self.cls_head = ClassificationHead(label_category_dict, 512*3 + 3)  # Feature size post-attention
        self.pooling = GlobalMaxAvgPool()

    def forward(self, t2w, dwi, adc, cln_var):
        # Encode each sequnece
        f_t2w = self.pooling(self.encoder_t2w(t2w)[-1])
        f_dwi = self.pooling(self.encoder_dwi(dwi)[-1])
        f_adc = self.pooling(self.encoder_adc(adc)[-1])

        # Cross-attention among sequences
        f_t2w_attention, f_dwi_attention, f_adc_attention = self.cross_attention(f_t2w, f_dwi, f_adc)
        
        # Concatenate or integrate the outputs
        concatenated_features = torch.cat([f_t2w_attention, f_dwi_attention, f_adc_attention], dim=1)
        concatenated_features = torch.cat([concatenated_features, cln_var], dim=1)

        # Classification
        logits = self.cls_head(concatenated_features)
        return logits

   
 
device = "cuda:0"
if __name__ == '__main__':
    label_category_dict = dict(binary_task=1)
    model = Model(in_channel=1, label_category_dict=label_category_dict, dim=3)
    model = model.to(device)

epochs = 300
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-5)
l1_lamda = 1e-5


def tensor_to_list(tensor):
    """Convert a tensor to a list, handling scalars properly."""
    if tensor.dim() == 0:  # Check if the tensor is a scalar
        return [tensor.item()]
    return tensor.tolist()

lowest_val_loss = float('inf')
# Initialize lists to store metric history
train_acc_history = []
train_loss_history = []
val_acc_history = []
val_loss_history = []
train_auc_history = []
val_auc_history = []
train_precision_history = []
train_recall_history = []
val_precision_history = []
val_recall_history = []
train_specificity_history = []
val_specificity_history = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    epoch_auc = 0
    all_preds = []
    all_labels = []
    train_cases = []
    all_prob_preds = []          # For training

    
    for t2w, dwi, adc, cln_var, label, file_path in tqdm(train_loader):
        # Move data to device
        t2w = t2w.to(device).float()
        dwi = dwi.to(device).float()
        adc = adc.to(device).float()
        cln_var = cln_var.to(device).float()
        label = label.to(device).float()

        # Forward pass through the model
        output = model(t2w, dwi, adc, cln_var)
        loss = criterion(output['binary_task'], label.unsqueeze(-1))
        
        
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lamda * l1_norm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.sigmoid(output['binary_task']).round()
        all_prob_preds += tensor_to_list(torch.sigmoid(output['binary_task']).squeeze().detach().cpu())  # For AUC
        all_preds += tensor_to_list(preds.squeeze().detach().cpu())
        all_labels += tensor_to_list(label.detach().cpu())

        acc = (preds.squeeze() == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        
       
    conf_mat = confusion_matrix(all_labels, all_preds, labels=[0,1])

    train_auc = roc_auc_score(all_labels, all_prob_preds)
    train_f1 = f1_score(all_labels, all_preds)
    # Calculate precision and recall
    train_precision = precision_score(all_labels, all_preds)
    train_recall = recall_score(all_labels, all_preds)
    
    # Calculate Specificity
    TN = conf_mat[0, 0]
    FP = conf_mat[0, 1]
    train_specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # Added check to avoid division by zero


    train_loss_history.append(epoch_loss.item())
    
    train_acc_history.append(epoch_accuracy.item())
    train_auc_history.append(train_auc)
    train_precision_history.append(train_precision)
    train_recall_history.append(train_recall)
    train_specificity_history.append(train_specificity)
    model.eval()
    #with torch.no_grad():
    with torch.inference_mode():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        epoch_val_auc = 0
        
        all_val_preds = []
        all_val_labels = []
        valid_cases = []
        all_val_prob_preds = []  # Added for AUC

        for t2w, dwi, adc, cln_var, label, file_path in tqdm(valid_loader):
            t2w = t2w.to(device).float()
            dwi = dwi.to(device).float()
            adc = adc.to(device).float()
            cln_var = cln_var.to(device).float()
            label = label.to(device).float()

            # Forward pass through the model
            val_output = model(t2w, dwi, adc, cln_var)
            val_loss = criterion(val_output['binary_task'], label.unsqueeze(-1))
            
            preds = torch.sigmoid(val_output['binary_task']).squeeze().round()
            #include probabilities to the output 
            all_val_prob_preds += tensor_to_list(torch.sigmoid(val_output['binary_task']).squeeze().detach().cpu())  # For AUC
            # Handling potential scalar output for preds
            pred_tensor = preds.detach().cpu()
            if pred_tensor.nelement() == 1:
                all_val_preds.append(pred_tensor.item())
            else:
                all_val_preds += list(pred_tensor.numpy())

            all_val_labels += list(label.detach().cpu().numpy())


            acc = (preds.squeeze() == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)
         
        val_conf_mat = confusion_matrix(all_val_labels, all_val_preds, labels=[0,1])

    
    epoch_val_auc = roc_auc_score(all_val_labels, all_val_prob_preds)
    val_f1 = f1_score(all_val_labels, all_val_preds)
    
    # Calculate validation precision and recall
    val_precision = precision_score(all_val_labels, all_val_preds)
    val_recall = recall_score(all_val_labels, all_val_preds)
    
    # Calculate Specificity
    TN = val_conf_mat[0, 0]
    FP = val_conf_mat[0, 1]
    val_specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # Added check to avoid division by zero

    
    val_loss_history.append(epoch_val_loss.item())
    val_acc_history.append(epoch_val_accuracy.item())
    val_auc_history.append(epoch_val_auc)
    val_precision_history.append(val_precision)
    val_recall_history.append(val_recall)
    val_specificity_history.append(val_specificity)

    print(
        f"Epoch: {epoch+1} - loss: {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - "
        f"AUC: {train_auc:.4f} - F1: {train_f1:.4f} - precision: {train_precision:.4f} - recall: {train_recall:.4f} - specificity: {train_specificity:.4f}\n - val_loss: {epoch_val_loss:.4f} - "
        f"val_acc: {epoch_val_accuracy:.4f} - val_AUC: {epoch_val_auc:.4f} - val_F1: {val_f1:.4f} - val_precision: {val_precision:.4f} - val_recall: {val_recall:.4f} - val_specificity: {val_specificity:.4f}\n"
    )
    # Save the model when the validation accuracy is the highest
    if epoch_val_loss < lowest_val_loss:
       lowest_val_loss = epoch_val_loss
       best_val_accuracy = epoch_val_accuracy
       best_epoch = epoch + 1
       best_confusion_matrix = val_conf_mat
       best_val_loss =  epoch_val_loss
       best_val_f1 = val_f1
       best_val_auc = epoch_val_auc
       best_val_precision = val_precision
       best_val_recall = val_recall
       best_val_specificity = val_specificity
       torch.save(model.state_dict(), f'/vgg_bi_parametric_model_Prostate_3D_{epoch+1}.pt')
       #torch.save(optimizer.state_dict(), f'/optimizer_epoch_{epoch+1}.pt')
 
       
       # Save variables to a txt file
    with open("", "w") as file:
        file.write("********Train******************\n")
        file.write(f"Highest Training Accuracy: {epoch_accuracy}\n")
        file.write(f"Train Confusion Matrix: {conf_mat}\n")
        file.write(f"Train Loss: {epoch_loss}\n")                                          
        file.write(f"Train F1 Score: {train_f1}\n")
        file.write(f"Train AUC: {train_auc}\n")
        file.write(f"Train Precision: {train_precision}\n")
        file.write(f"Train Recall: {train_recall}\n")
        file.write(f"Train  Specificity: {train_specificity}\n")
        file.write("********Val******************\n")
        file.write(f"best_epoch: {best_epoch}\n")
        file.write(f"Highest Validation Accuracy: {epoch_val_accuracy}\n")
        file.write(f"Best Confusion Matrix: {best_confusion_matrix}\n")
        file.write(f"Best Validation Loss: {best_val_loss}\n")
        file.write(f"Best Validation F1 Score: {best_val_f1}\n")
        file.write(f"Best Validation AUC: {best_val_auc}\n")
        file.write(f"Best Validation Precision: {best_val_precision}\n")
        file.write(f"Best Validation Recall: {best_val_recall}\n")
        file.write(f"Best Validation Specificity: {best_val_specificity}\n") 
        
     # Plotting the metrics
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Adjust the size as needed

    # Training Loss
    axs[0, 0].plot(train_loss_history, label='Training Loss')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')

    # Training Accuracy
    axs[0, 1].plot(train_auc_history, label='Training AUC')
    axs[0, 1].set_title('Training AUC')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('AUC')

    # Validation Loss
    axs[1, 0].plot(val_loss_history, label='Validation Loss')
    axs[1, 0].set_title('Validation Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')

    # Validation Accuracy
    axs[1, 1].plot(val_auc_history, label='Validation AUC')
    axs[1, 1].set_title('Validation AUC')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('AUC')

    for ax in axs.flat:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('/training_validation_metrics.png', dpi=300)  # Save figure to the current directory
    plt.close(fig)  # Close the figure to free memory
    
    # Plotting precision and recall in a separate figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Precision plot
    axs[0].plot(train_precision_history, label='Training Precision')
    axs[0].plot(val_precision_history, label='Validation Precision')
    axs[0].set_title('Precision')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Precision')
    axs[0].legend()
    axs[0].grid(True)
    
    # Recall plot
    axs[1].plot(train_recall_history, label='Training Recall')
    axs[1].plot(val_recall_history, label='Validation Recall')
    axs[1].set_title('Recall')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Recall')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/precision_recall_metrics.png', dpi=300)  # Save the precision and recall figure
    plt.close(fig)
    
    
    # Plotting specificity in a separate figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    
    # specificity training plot
    axs[0].plot(train_specificity_history, label='Training Specificity')
    axs[0].set_title('Specificity')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()
    axs[0].grid(True)
    
    # specificity validation plot
    axs[1].plot(val_specificity_history, label='Validation Specificity')
    axs[1].set_title('Specificity')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/specificity_metrics.png', dpi=300)  # Save the precision and recall figure
    plt.close(fig)


