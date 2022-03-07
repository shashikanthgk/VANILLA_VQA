import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ImgEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ImgEncoder, self).__init__()
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[-1].in_features  
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1])    
        self.model = model                             
        self.fc = nn.Linear(in_features, embed_size)    

    def forward(self, image):
        with torch.no_grad():
            img_feature = self.model(image)                  
        img_feature = self.fc(img_feature)                   
        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)            

        return img_feature


class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     

    def forward(self, question):
        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature


class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size):

        super(VqaModel, self).__init__()
        self.img_encoder = ImgEncoder(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, img, qst):

        img_feature = self.img_encoder(img)                     
        qst_feature = self.qst_encoder(qst)                     
        combined_feature = torch.mul(img_feature, qst_feature)  
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)          

        return combined_feature

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
        img_feature = self.fc(img_feature)                          
        return img_feature


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
        img_feature = img_feature.view(-1, 512, 196).transpose(1,2) 
        img_feature = self.fc(img_feature)                          
        return img_feature


class Attention(nn.Module):
    def __init__(self, num_channels, embed_size, dropout=True):
        super(Attention, self).__init__()
        self.ff_image = nn.Linear(embed_size, num_channels)
        self.ff_questions = nn.Linear(embed_size, num_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(num_channels, 1)
    
    def forward(self, vi, vq):
        hi = self.ff_image(vi)                      #[batch_size,196,512]
        hq = self.ff_questions(vq).unsqueeze(dim=1) #[batch_size,1,512]
        ha = torch.tanh(hi+hq)                      #[batch_size,196,512]
        if self.dropout:
            ha = self.dropout(ha)       
        ha = self.ff_attention(ha)                  #[batch_size,196,1]
        pi = torch.softmax(ha, dim=1)               #[batch_size,196,1]
        self.pi = pi
        vi_attended = (pi * vi).sum(dim=1)          #[batch_size,1024]
        u = vi_attended + vq                        #[batch_size,1024]
        return u

class VWSA(nn.Module):
    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size): 
        super(VWSA, self).__init__()
        self.num_mlp_layer = 1
        self.img_encoder = ImgAttentionEncoder(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.att = Attention(512, embed_size)
        self.tanh = nn.Tanh()
        self.mlp = nn.Sequential(nn.Dropout(p=0.5),
                            nn.Linear(embed_size, ans_vocab_size))
        self.attn_features = [] 

    def forward(self, img, qst):
        img_feature = self.img_encoder(img)                    
        qst_feature = self.qst_encoder(qst)                     
        vi = img_feature
        u = qst_feature
        u = self.att(vi, u)         
        combined_feature = self.mlp(u)
        return combined_feature



class SAN(nn.Module):
    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size,num_attention_layer,device): 
        super(SAN, self).__init__()
        self.num_attention_layer = num_attention_layer
        self.num_mlp_layer = 1
        self.num_attention_layer = num_attention_layer
        # self.img_encoder = ImgAttentionEncoder(embed_size)
        self.img_encoder = ImgFeatureFusionEncoder(embed_size,device)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.san = nn.ModuleList([Attention(512, embed_size)]*self.num_attention_layer)
        self.tanh = nn.Tanh()
        self.mlp = nn.Sequential(nn.Dropout(p=0.5),
                            nn.Linear(embed_size, ans_vocab_size))
        self.attn_features = []  ## attention features


    def forward(self, img, qst):
        img_feature = self.img_encoder(img)                    
        qst_feature = self.qst_encoder(qst)                     
        vi = img_feature
        u = qst_feature
        for attn_layer in self.san:
            u = attn_layer(vi, u)
            # self.attn_features.append(attn_layer.pi)
        combined_feature = self.mlp(u)
        return combined_feature

