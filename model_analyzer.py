
from base64 import encode
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from sklearn.preprocessing import minmax_scale
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





class ImgAttentionEncoder(nn.Module):

    def __init__(self, embed_size):

        super(ImgAttentionEncoder, self).__init__()
        vggnet_feat = models.vgg19(pretrained=True).features
        modules = list(vggnet_feat.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.fc = nn.Sequential(nn.Linear(self.cnn[-3].out_channels, embed_size),
                                nn.Tanh())     

    def forward(self, image):
        with torch.no_grad():
            img_feature = self.cnn(image)
        return img_feature
        img_feature = img_feature.view(-1, 512, 196).transpose(1,2) 
        return img_feature

class ImgFeatureFusionEncoder2(nn.Module):

    def __init__(self, embed_size,device):
        super(ImgFeatureFusionEncoder2, self).__init__()
        vggnet_feat = models.vgg19(pretrained=True).features
        modules1 = list(vggnet_feat.children())[:18]
        modules2 = list(vggnet_feat.children())[18:27]
        modules3 = list(vggnet_feat.children())[27:-2]
        self.cnn1 = nn.Sequential(*modules1)
        self.cnn2 = nn.Sequential(*modules2)
        self.cnn3 = nn.Sequential(*modules3)
        self.device = device
        self.fc = nn.Sequential(nn.Linear(self.cnn3[-3].out_channels, embed_size),
                                nn.Tanh())     

    def forward(self, image):
        with torch.no_grad():
            feature_layer_1 = self.cnn1(image) # 256,14,14
            feature_layer_2 = self.cnn2(feature_layer_1) # 512,28,28
            feature_layer_3 = self.cnn3(feature_layer_2) # 512,14,14

        convoluted_1 = nn.Conv2d(256,512,1, bias=False).to(self.device)(feature_layer_1)
        # pooled_layer1 = nn.MaxPool2d(4, stride=4)(convoluted_1).to(self.device)
        # pooled_layer2 = nn.MaxPool2d(2, stride=2)(feature_layer_2)
        pooled_layer1 = nn.AvgPool2d(4, stride=4)(convoluted_1).to(self.device)
        pooled_layer2 = nn.AvgPool2d(2, stride=2)(feature_layer_2)
        p_12_layer = torch.add(pooled_layer1,pooled_layer2)
        fused_feature = torch.add(p_12_layer,feature_layer_3)
        img_feature = fused_feature.view(-1, 512, 196).transpose(1,2) 
        return img_feature

class ImgFeatureFusionEncoder(nn.Module):

    def __init__(self, embed_size,device):

        super(ImgFeatureFusionEncoder, self).__init__()
        vggnet_feat = models.vgg19(pretrained=True).features
        modules1 = list(vggnet_feat.children())[:18]
        modules2 = list(vggnet_feat.children())[18:27]
        modules3 = list(vggnet_feat.children())[27:-2]
        self.cnn1 = nn.Sequential(*modules1)
        self.cnn2 = nn.Sequential(*modules2)
        self.cnn3 = nn.Sequential(*modules3)
        self.device = device
        self.fc = nn.Sequential(nn.Linear(self.cnn3[-3].out_channels, embed_size),
                                nn.Tanh())     

    def forward(self, image):
        with torch.no_grad():
            feature_layer_1 = self.cnn1(image)
            feature_layer_2 = self.cnn2(feature_layer_1)
            feature_layer_3 = self.cnn3(feature_layer_2)

        convoluted_1 = nn.Conv2d(256,512,1, bias=False).to(self.device)(feature_layer_1)
        # pooled_layer1 = nn.MaxPool2d(4, stride=4)(convoluted_1).to(self.device)
        # pooled_layer2 = nn.MaxPool2d(2, stride=2)(feature_layer_2)
        pooled_layer1 = nn.AvgPool2d(4, stride=4)(convoluted_1).to(self.device)
        pooled_layer2 = nn.AvgPool2d(2, stride=2)(feature_layer_2)
        p_12_layer = torch.add(pooled_layer1,pooled_layer2)
        fused_feature = torch.add(p_12_layer,feature_layer_3)
        img_feature = fused_feature.view(-1, 512, 196).transpose(1,2) 
        return img_feature



class ImgFeatureFusionEncoder3(nn.Module):

    def __init__(self, embed_size,device):

        super(ImgFeatureFusionEncoder3, self).__init__()
        vggnet_feat = models.vgg19(pretrained=True).features
        modules1 = list(vggnet_feat.children())[:18]
        modules2 = list(vggnet_feat.children())[18:27]
        modules3 = list(vggnet_feat.children())[27:-2]
        self.cnn1 = nn.Sequential(*modules1)
        self.cnn2 = nn.Sequential(*modules2)
        self.cnn3 = nn.Sequential(*modules3)
        self.device = device
        self.fc = nn.Sequential(nn.Linear(self.cnn3[-3].out_channels, embed_size),
                                nn.Tanh())     

    def forward(self, image):
        with torch.no_grad():
            feature_layer_1 = self.cnn1(image)
            feature_layer_2 = self.cnn2(feature_layer_1)
            feature_layer_3 = self.cnn3(feature_layer_2)

        convoluted_1 = nn.Conv2d(256,512,1, bias=False).to(self.device)(feature_layer_1)
        pooled_layer1 = nn.MaxPool2d(4, stride=4)(convoluted_1).to(self.device)
        pooled_layer2 = nn.MaxPool2d(2, stride=2)(feature_layer_2)
        # pooled_layer1 = nn.AvgPool2d(4, stride=4)(convoluted_1).to(self.device)
        # pooled_layer2 = nn.AvgPool2d(2, stride=2)(feature_layer_2)
        p_12_layer = torch.add(pooled_layer1,pooled_layer2)
        fused_feature = torch.add(p_12_layer,feature_layer_3)
        img_feature = fused_feature.view(-1, 512, 196).transpose(1,2) 
        return img_feature





class SAN(nn.Module):
    def __init__(self,attentionEncoder): 
        super(SAN, self).__init__()
        self.img_encoder = attentionEncoder
        # self.img_encoder = ImgFeatureFusionEncoder(embed_size,device)
    def forward(self, img):
        img_feature = self.img_encoder(img)                    
        return img_feature



def image_loader(loader, image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

# data_transforms = transforms.Compose([transforms.ToTensor(),
#                     transforms.Normalize((0.485, 0.456, 0.406),
#                                         (0.229, 0.224, 0.225))]) 

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
    
])



def plot_image(image_tensor):
    f, axarr = plt.subplots(23,23)
    row = 0
    col = 0

    for i in range(image_tensor[0]):
        axarr[row,col].imshow(image_tensor[i])
        col += 1
        if col == 23:
            row += 1
            col = 0
    plt.show()

def train_model(encoder,image,name):
    model_ft = SAN(attentionEncoder=encoder).to(device)
    model_ft.eval()
    x = model_ft(image)
    x = x.detach().cpu().numpy()[0]
    shape = x.shape
    print(shape)
    fig, axes = plt.subplots(23, 23)
    count = 1
    for idx, arch in enumerate(x):
        i = (idx % 23)
        j = (idx // 23)
        shape = arch.shape 
        image_scaled = minmax_scale(arch.ravel(), feature_range=(0,1)).reshape(shape)
        img = Image.fromarray(np.uint8(image_scaled * 255) , 'L')
        img = np.asarray(img)
        axes[i, j].imshow(img,cmap='gray',interpolation='none')
        axes[i,j].axis('off')
        # axes[i, j].set_cmap('hot')
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    # image_scaled = minmax_scale(x.ravel(), feature_range=(0,1)).reshape(shape)
    # img = Image.fromarray(np.uint8(image_scaled * 255) , 'L')
    # img.save("ResultImages/"+name+".PNG")
    # print(image_scaled)
    # plt.imsave("ResultImages/"+name+".jpg",image_scaled)



img = image_loader(data_transforms,"final.jpg").to(device)
normal_encoder = ImgAttentionEncoder(1024)
# fusion_encoder_avg = ImgFeatureFusionEncoder(1024,device=device)
# fusion_encoder_max = ImgFeatureFusionEncoder3(1024,device); 
# fusion_encoder_concat = ImgFeatureFusionEncoder2(1024,device); 
# print(normal_encoder.shape)
train_model(normal_encoder,img,"normal_encoder")
# train_model(fusion_encoder_avg,img,"fusion_encoder_avg")
# train_model(fusion_encoder_max,img,"fusion_encoder_max")
# train_model(fusion_encoder_concat,img,"fusion_encoder_concat")
