# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:29:25 2024

@author: Yash Chonkar
"""

import yolov5
from pathlib import Path
import os

import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data import  TensorDataset
import numpy as np
from yolo import YOLO
import cv2
import carla
'''
'''

#Activation Function
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# Basic Convolution Block with Mish 
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()
        '''CONV + BATCHNORM + MISH'''
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

# Residual Block 
class ResidualBlock(nn.Module):
    def __init__(self, channels, hidden_channels=None, residual_activation=nn.Identity()):
        super(ResidualBlock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels // 2

        self.block = nn.Sequential(BasicConv(channels, hidden_channels, kernel_size=1), BasicConv(hidden_channels, channels, kernel_size=3))

    def forward(self, x):
        return x + self.block(x)

# YOLO framework 
class YOLO(nn.Module):
    def __init__(self, num_classes, max_objects=10):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        self.max_objects = max_objects
        
        # Backbone using BasicConv with Mish 
        self.conv1 = BasicConv(3, 32, 3, 1) 
        self.conv2 = BasicConv(32, 64, 3, 2) 
        self.res1 = ResidualBlock(64)        
        
        self.conv3 = BasicConv(64, 128, 3, 2) # Downsampling, 128 channels
        self.res2 = ResidualBlock(128)        
        
        self.conv4 = BasicConv(128, 256, 3, 2) # Downsampling, 256 channels
        self.res3 = nn.Sequential( *[ResidualBlock(256) for _ in range(4)]  )
        
        self.conv5 = BasicConv(256, 512, 3, 2) # Downsampling, 512 channels
        self.res4 = nn.Sequential(*[ResidualBlock(512) for _ in range(4)] )
        
        self.conv6 = BasicConv(512, 1024, 3, 2) # Downsampling, 1024 channels
        self.res5 = nn.Sequential( *[ResidualBlock(1024) for _ in range(2)]  )

        # Detection head 
        self.conv7 = BasicConv(1024, 512, 3, 1)
        self.conv8 = BasicConv(512, 1024, 3, 1)
        self.conv9 = nn.Conv2d(1024, self.max_objects * 5, 1, 1, 0)  # Output channels: max_objects * 5

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        
        x = self.conv3(x)
        x = self.res2(x)
        
        x = self.conv4(x)
        x = self.res3(x)
        
        x = self.conv5(x)
        x = self.res4(x)
        
        x = self.conv6(x)
        x = self.res5(x)
        
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)  # Final detection output
        
        # Reshape  output to match label shape: [batch_size, max_objects, 5]
        batch_size = x.size(0)
        num_channels = x.size(1)
        height = x.size(2)
        width = x.size(3)

        expected_channels = self.max_objects * 5
        if num_channels != expected_channels:
            raise ValueError(f"Mismatch in number of channels: {num_channels} vs {expected_channels}")

        x = x.view(batch_size, self.max_objects, 5, height * width)
        # Rearranging dimensions to [batch_size, max_objects, height * width, 5]
        x = x.permute(0, 1, 3, 2)

        return x

    def yolo_loss(self, outputs, labels):
        # Print shapes for debugging
        #print(f"Outputs shape: {outputs.shape}")
        #print(f"Labels shape: {labels.shape}")
    
        # Extract dimensions
        batch_size = outputs.size(0)
        num_objects = outputs.size(1)
        
        # average to remove dimension
        outputs = outputs.mean(dim=2)  # Now outputs shape will be [batch_size, num_objects, 5]
    
        # Reshape outputs to [batch_size * num_objects, 5]
        outputs = outputs.view(batch_size * num_objects, -1)
        
        labels = labels.view(batch_size * num_objects, -1)
    
        # Calculate loss
        criterion = nn.MSELoss()  # MSE loss
        loss = criterion(outputs, labels)
        return loss

    def train_model(self, images, labels, num_epochs=10, lr=0.001, weight_decay=1e-4):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    
        images = torch.stack(images).to(device)
        labels = torch.stack(labels).to(device)
    
        if images.size(0) != labels.size(0):
            raise ValueError("Mismatch between number of images and labels")
    
        # Training loop
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
    
            for i in range(images.size(0)):
                batch_image = images[i].unsqueeze(0)  # Add batch dimension for single image
                batch_label = labels[i].unsqueeze(0)  # Add batch dimension for single label
                
                optimizer.zero_grad()
                
                outputs = self(batch_image)
                loss = self.yolo_loss(outputs, batch_label)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
    
            epoch_loss = running_loss / images.size(0)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
        return self




class YoloDataset:
    def __init__(self,image_directory_list,label_directory_list):
        image_folder = os.path.join(*image_directory_list)
        image_files = os.listdir(image_folder)
        transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize 
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.image_list = []
     
        for img_file in image_files:
            # Construct the full path to the image
            img_path = os.path.join(image_folder, img_file)
            
            # convert image to RGB and transform
            img = Image.open(img_path).convert('RGB')  
            img_tensor = transform(img)
            
            self.image_list.append(img_tensor)
           
        label_folder = os.path.join(*label_directory_list)
        label_files = os.listdir(label_folder)
        self.label_list = []
        max_objects = 10  # Maximum number of objects expected to detect
  
        for label_file in label_files:
            # Full path to label file
            label_path = os.path.join(label_folder, label_file)
            
            with open(label_path, 'r') as file:
                lines = file.readlines()
                
                #intial tensor filled with zeroes
                label_tensor = torch.zeros((max_objects, 5))
                #if file not empty
                if lines:
                    for i, line in enumerate(lines):
                        if i >= max_objects:  
                            break
                        parts = line.strip().split()
                        if len(parts) == 5:
                            label_tensor[i] = torch.tensor([float(x) for x in parts])
            
            # Append the tensor to the list
            self.label_list.append(label_tensor)
            
            
            
    def get_data(self):
        return self.image_list, self.label_list
    
    def __len__(self):
        return len(self.image_list)  
           
           
         
        
            
            
        







transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to a consistent size, if needed
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize, if needed
])




#Load data and train
image_directory_list =  ['carla-object-detection-dataset','images','train']
label_directory_list =  ['carla-object-detection-dataset','yolo_labels','train']
dataset = YoloDataset(image_directory_list, label_directory_list)
images, labels = dataset.get_data()

num_classes = 5
model = YOLO(num_classes=num_classes, max_objects=10)
trained_model = model.train_model(images, labels, num_epochs=10, lr=0.001)



torch.save(trained_model.state_dict(), 'yolo_model.pth')

def calculate_jaccard(boxes1, boxes2):
    # Convert from center-xy and width-height to corner coordinates (x1, y1, x2, y2)
    boxes1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    boxes1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    boxes1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    boxes1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2

    boxes2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    boxes2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    boxes2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2

    # Prepare the reformatted boxes (x1, y1, x2, y2)
    reformatted_boxes1 = torch.zeros_like(boxes1)
    reformatted_boxes2 = torch.zeros_like(boxes2)

    reformatted_boxes1[:, 0], reformatted_boxes1[:, 1] = boxes1_x1, boxes1_y1
    reformatted_boxes1[:, 2], reformatted_boxes1[:, 3] = boxes1_x2, boxes1_y2

    reformatted_boxes2[:, 0], reformatted_boxes2[:, 1] = boxes2_x1, boxes2_y1
    reformatted_boxes2[:, 2], reformatted_boxes2[:, 3] = boxes2_x2, boxes2_y2

    num_boxes1 = reformatted_boxes1.size(0)
    num_boxes2 = reformatted_boxes2.size(0)

    # Calculate the intersection coordinates
    top_left = torch.max(reformatted_boxes1[:, :2].unsqueeze(1).expand(num_boxes1, num_boxes2, 2),reformatted_boxes2[:, :2].unsqueeze(0).expand(num_boxes1, num_boxes2, 2))
    bottom_right = torch.min(reformatted_boxes1[:, 2:].unsqueeze(1).expand(num_boxes1, num_boxes2, 2),reformatted_boxes2[:, 2:].unsqueeze(0).expand(num_boxes1, num_boxes2, 2))

    # Compute the intersection area
    intersection = torch.clamp((bottom_right - top_left), min=0)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]

    # Compute the union area
    area1 = ((reformatted_boxes1[:, 2] - reformatted_boxes1[:, 0]) *(reformatted_boxes1[:, 3] - reformatted_boxes1[:, 1])).unsqueeze(1).expand_as(intersection_area)
    area2 = ((reformatted_boxes2[:, 2] - reformatted_boxes2[:, 0]) *(reformatted_boxes2[:, 3] - reformatted_boxes2[:, 1])).unsqueeze(0).expand_as(intersection_area)

    union_area = area1 + area2 - intersection_area

    # Return the Intersection over Union (IoU) for each box pair
    return intersection_area / union_area



def augment_single_image(self, annotation_line, input_shape, hue_adjust=.1, sat_adjust=1.5, val_adjust=1.5):
    '''Applies augmentations to image, expand training set.'''
    height, width = input_shape
    image_data = []
    box_data = []
    # get images and boxes
    line_content = annotation_line.split()
    image = Image.open(line_content[0]).convert("RGB")
    original_width, original_height = image.size
    boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])
    
    # Random horizontal flip
    if np.random.rand() < 0.5 and len(boxes) > 0:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        boxes[:, [0, 2]] = original_width - boxes[:, [2, 0]]
    # Random rotation (-15 and 15 deg)
    angle = np.random.uniform(-15, 15)
    image = image.rotate(angle, expand=True)
    # Random scaling
    scaling_factor = np.random.uniform(0.8, 1.2)
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    image = image.resize((new_width, new_height), Image.BICUBIC)
    # Adjust bounding boxes 
    if len(boxes) > 0:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * new_width / original_width
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * new_height / original_height
    # Random color distortion
    hue_shift = np.random.uniform(-hue_adjust, hue_adjust)
    sat_scale = np.random.uniform(1, sat_adjust) if np.random.rand() < 0.5 else 1 / np.random.uniform(1, sat_adjust)
    val_scale = np.random.uniform(1, val_adjust) if np.random.rand() < 0.5 else 1 / np.random.uniform(1, val_adjust)
    # Convert
    hsv_image = rgb_to_hsv(np.array(image) / 255.0)
    hsv_image[..., 0] = (hsv_image[..., 0] + hue_shift) % 1.0
    hsv_image[..., 1] *= sat_scale
    hsv_image[..., 2] *= val_scale
    hsv_image = np.clip(hsv_image, 0, 1)
    augmented_image = Image.fromarray((hsv_to_rgb(hsv_image) * 255).astype(np.uint8))
    # Adjust bounding boxes to fit dimensions
    if len(boxes) > 0:
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, new_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, new_height)

    return augmented_image, boxes



def process_rgb_image(image_data, tag):
    image_array = np.array(image_data.raw_data)
    reshaped_img = image_array.reshape((480, 360, 4))
    img_rgb = reshaped_img[:, :, :3]
    # Convert image to PIL format and run YOLO 
    image_pil = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    detected_img = yolo.detect_image(image_pil)
    img_rgb_with_obj = cv2.cvtColor(np.asarray(detected_img), cv2.COLOR_RGB2BGR)
    cv2.imshow(tag, img_rgb)
    cv2.imshow('Detected Objects', img_rgb_with_obj)
    cv2.waitKey(1)

    return img_rgb / 255.0
