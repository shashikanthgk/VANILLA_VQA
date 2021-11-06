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


class ImgAttentionEncoder(nn.Module):

    def __init__(self, embed_size):

        super(ImgAttentionEncoder, self).__init__()
        vggnet_feat = models.vgg19(pretrained=True).features
        modules = list(vggnet_feat.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.fc = nn.Sequential(nn.Linear(self.cnn[-3].out_channels, embed_size),
                                nn.Tanh())     # feature vector of image

    def forward(self, image):
        with torch.no_grad():
            img_feature = self.cnn(image)                           # [batch_size, vgg16(19)_fc=4096]
        img_feature = img_feature.view(-1, 512, 196).transpose(1,2) # [batch_size, 196, 512]
        img_feature = self.fc(img_feature)                          # [batch_size, 196, embed_size]
        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)        
        return img_feature


class Attention(nn.Module):
    def __init__(self, num_channels, embed_size, dropout=True):
        super(Attention, self).__init__()
        self.ff_image = nn.Linear(embed_size, num_channels)
        self.ff_questions = nn.Linear(embed_size, num_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(num_channels, 1)
    
    def forward(self, vi, vq):
        # N * 196 * 1024 -> N * 196 * 512
        hi = self.ff_image(vi)
        # N * 1024 -> N * 512 -> N * 1 * 512
        hq = self.ff_questions(vq).unsqueeze(dim=1)
        # N * 196 * 512
        ha = F.tanh(hi + hq)
        if getattr(self, 'dropout'):
            ha = self.dropout(ha)
        # N * 196 * 512 -> N * 196 * 1 -> N * 196
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = F.softmax(ha)
        # (N * 196 * 1, N * 196 * 1024) -> N * 1024
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq
        return u

    # def forward(self, vi, vq):
    #     hi = self.ff_image(vi)
    #     hq = self.ff_questions(vq).unsqueeze(dim=1)
    #     ha = torch.tanh(hi+hq)
    #     if self.dropout:
    #         ha = self.dropout(ha)
    #     ha = self.ff_attention(ha)
    #     pi = torch.softmax(ha, dim=1)
    #     self.pi = pi
    #     vi_attended = (pi * vi).sum(dim=1)
    #     u = vi_attended + vq
    #     return u

class VWSA(nn.Module):
    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size): 
        super(VWSA, self).__init__()
        self.num_mlp_layer = 1
        self.img_encoder = ImgAttentionEncoder(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.att = Attention(512, embed_size)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)
        self.dropout = nn.Dropout(0.5)
        # self.mlp = nn.Sequential(nn.Dropout(p=0.5),
        #                     nn.Linear(embed_size, ans_vocab_size))
        self.attn_features = []  ## attention features

    def forward(self, img, qst):
        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
        vi = img_feature
        u = qst_feature
        u = self.att(vi, u)          
        # combined_feature = self.mlp(u)
        combined_feature = self.tanh(u)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)          
        return combined_feature