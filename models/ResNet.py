from __future__ import absolute_import

import os
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision

__all__ = ['ResNet50TP', 'ResNet50TA', 'ResNet50RNN']

class ResNet50TP(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TP, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, z, use_depth=False):
        if not use_depth:
            #senza Depth - Codice Originale
            b = x.size(0)
            t = x.size(1)
            x = x.view(b*t,x.size(2), x.size(3), x.size(4))
            x = self.base(x)
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(b,t,-1)
            x=x.permute(0,2,1)
            f = F.avg_pool1d(x,t)
            f = f.view(b, self.feat_dim)
            if not self.training:
                return f
            y = self.classifier(f)

            if self.loss == {'xent'}:
                return y
            elif self.loss == {'xent', 'htri'}:
                return y, f, None
            elif self.loss == {'cent'}:
                return y, f, None
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))
        else:
            b = x.size(0)
            t = x.size(1)
            bd = z.size(0)
            td = z.size(1)

            # Rete Depth
            z = z.view(bd * td, z.size(2), z.size(3), z.size(4))
            z1 = self.base(z)
            # print("z1: ", z1.shape)
            z2 = F.avg_pool2d(z1, z1.size()[2:])  # avg pool non ha return_indices
            z3 = z2.view(bd, td, -1)
            z4 = z3.permute(0, 2, 1)
            fd = F.avg_pool1d(z4, td)
            fd2 = fd.view(bd, self.feat_dim)

            x = x.view(b * t, x.size(2), x.size(3), x.size(4))
            x1 = self.base(x)
            x1 = torch.add(x1, z1)  # FUSION 1
            x2 = F.avg_pool2d(x1, x1.size()[2:])
            x2 = torch.add(x2, z2)  # FUSION 2
            x3 = x2.view(b, t, -1)
            x4 = x3.permute(0, 2, 1)
            x4 = torch.add(x4, z4)  # FUSION 3
            f = F.avg_pool1d(x4, t)
            f = f.view(b, self.feat_dim)
            f = torch.add(f, fd2)   # FUSION 4

            if not self.training:
                return f
            y = self.classifier(f)

            if self.loss == {'xent'}:
                return y
            elif self.loss == {'xent', 'htri'}:
                return y, f, None
            elif self.loss == {'cent'}:
                return y, f, None
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))

'''
class ResNet50TP_Massimo(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TP, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, z, use_depth=False):
        #print(x.shape, z.shape)
        #print("x: ", x.size(0), x.size(1), x.size(2), x.size(3), x.size(4))
        #print("z: ", z.size(0), z.size(1), z.size(2), z.size(3), z.size(4))
        #print("---")

        b = x.size(0)
        t = x.size(1)
        bd = x.size(0)
        td = x.size(1)

        # Rete Depth
        z = z.view(bd * td, z.size(2), z.size(3), z.size(4))
        z1 = self.base(z)
        #print("z1: ", z1.shape)
        z2 = F.avg_pool2d(z1, z1.size()[2:])  # avg pool non ha return_indices
        z3 = z2.view(bd, td, -1)
        z4 = z3.permute(0, 2, 1)
        fd = F.avg_pool1d(z4, td)
        fd2 = fd.view(bd, self.feat_dim)



        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x1 = self.base(x)
        #print("x1: ", x1.shape)
        if (x.size(0) == z1.size(0)) and use_depth:  # per qualche motivo alcune immagini non hanno dim uguali
            x1 = torch.add(x1, z1)

        x2 = F.avg_pool2d(x1, x1.size()[2:])
        if (x2.size(0) == z2.size(0)) and use_depth:  # per qualche motivo alcune immagini non hanno dim uguali
            x2 = torch.add(x2, z2)

        x3 = x2.view(b,t,-1)
        x4 = x3.permute(0,2,1)
        if (x4.size(0) == z4.size(0)) and use_depth:  # per qualche motivo alcune immagini non hanno dim uguali
            x4 = torch.add(x4, z4)

        f = F.avg_pool1d(x4,t)
        f = f.view(b, self.feat_dim)

        if (f.size(0) == fd2.size(0)) and use_depth:  # per qualche motivo alcune immagini non hanno dim uguali
            f = torch.add(f, fd2)

        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f, fd2
        elif self.loss == {'cent'}:
            return y, f, fd2
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
'''

'''
class ResNet50TP_2(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TP, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.classifier = nn.Linear(int(self.feat_dim/4), num_classes)


    def forward(self, x, z):
        print(x.shape,z.shape)
        print("x: ",x.size(0),x.size(1),x.size(2),x.size(3),x.size(4))
        print("z: ",z.size(0), z.size(1), z.size(2), z.size(3), z.size(4))
        print("---")
        b = x.size(0)
        t = x.size(1)
        bd = z.size(0)
        td = z.size(1)


        #Rete Depth
        z = z.view(bd*td,z.size(2), z.size(3), z.size(4))
        z1 = self.base(z)
        print("z1: ", z1.shape)
        z2 = F.avg_pool2d(z1, z1.size()[2:]) #avg pool non ha return_indices
        z3 = z2.view(bd,td,-1)
        z4 = z3.permute(0,2,1)
        fd = F.avg_pool1d(z4,td)
        fd2 = fd.view(bd*4, int(self.feat_dim/4))



        #Rete base RGB
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        print("x: ", x.shape)
        if(x.size(0)==z1.size(0)): #per qualche motivo alcune immagini non hanno dim uguali
            x = torch.add(x,z1)
        #else: altrimenti non fare nulla per ora, poi si vede
            #z1 = z1.view(1,-1)# si dovrebbe in qualche modo rendere z1 sommabile...
            #x = torch.add(x,z1) #e poi  dovrebbe andare
        x2 = F.avg_pool2d(x, x.size()[2:]) #avg pool non ha return_indices
        if(x2.size(0)==z2.size(0)): #per qualche motivo alcune immagini non hanno dim uguali
            x2 = torch.add(x2,z2)
        #else:
            #z2 = z2.view(1,-1)
            #x2 = torch.add(x2,z2) #in teoria ora dovrebbe andare
        x3 = x2.view(b,t,-1)
        x4 = x3.permute(0,2,1)
        if(x4.size(0)==z4.size(0)): #per qualche motivo alcune immagini non hanno dim uguali
            x2 = torch.add(x4,z4)
        f = F.avg_pool1d(x4,t)
        f = f.view(b*4, int(self.feat_dim/4))
        if(f.size(0)==fd2.size(0)): #per qualche motivo alcune immagini non hanno dim uguali
            f = torch.add(f,fd2)




        if not self.training:
            return f #rivedere questo
        y = self.classifier(f)
        #riaggiustiamo dimensioni
        f = f.view(16, -1)
        fd2 = fd2.view(16, -1)
        y = y.view(16,-1)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f, fd2
        elif self.loss == {'cent'}:
            return y, f, fd2
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TA_2(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax' # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048 # feature dimension
        self.middle_dim = 256 # middle layer dimension
        self.classifier = nn.Linear(self.feat_dim/4, num_classes)
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
    def forward(self, x, z):
        b = x.size(0)
        t = x.size(1)
        bd = z.size(0)
        td = z.size(1)
        #Depth
        z = z.view(bd*td, z.size(2), z.size(3), z.size(4))
        z1 = self.base(z)
        ad = F.relu(self.attention_conv(z1))
        ad = ad.view(bd, td, self.middle_dim)
        ad1 = ad.permute(0,2,1)
        ad = F.relu(self.attention_tconv(ad1))
        ad = ad.view(bd, td)
        z2 = F.avg_pool2d(z1, z1.size()[2:])

        #RGB
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        if(x.size(0)==z1.size(0)): #per qualche motivo alcune immagini non hanno dim uguali
            x = torch.add(x,z1)
        a = F.relu(self.attention_conv(x))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        if(a.size(0)==ad1.size(0)): #per qualche motivo alcune immagini non hanno dim uguali
            x = torch.add(x,ad1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if(x.size(0)==z2.size(0)): #per qualche motivo alcune immagini non hanno dim uguali
            x = torch.add(x,z2)


        if self. att_gen=='softmax':
            a = F.softmax(a, dim=1)
            ad = F.softmax(ad, dim=1)
        elif self.att_gen=='sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
            ad = F.sigmoid(ad)
            ad = F.normalize(ad, p=1, dim=1)
        else:
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim/4)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)

        z = z.view(bd, td, -1)
        ad = torch.unsqueeze(ad, -1)
        ad = ad.expand(bd, td, self.feat_dim/4)
        att_z = torch.mul(z,ad)
        att_z = torch.sum(att_z,1)

        f = att_x.view(b,self.feat_dim/4)
        fd = att_z.view(bd,self.feat_dim/4)

        if not self.training:
            return f, fd
        y = self.classifier(f)
        #riaggiustiamo dimensioni
        f = f.view(16, -1)
        fd = fd.view(16, -1)
        y = y.view(16,-1)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f, fd
        elif self.loss == {'cent'}:
            return y, f, fd
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
'''

class ResNet50TA(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim,
                                        #[7, 4])  # 7,4 cooresponds to 224, 112 input image size
                                        [4, 4])  # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

    def forward(self, x, z, use_depth=False):

        if not use_depth:
            #senza Depth - Codice Originale
            b = x.size(0)
            t = x.size(1)
            x = x.view(b * t, x.size(2), x.size(3), x.size(4))
            x = self.base(x)
            a = F.relu(self.attention_conv(x))
            a = a.view(b, t, self.middle_dim)
            a = a.permute(0, 2, 1)
            a = F.relu(self.attention_tconv(a))
            a = a.view(b, t)
            x = F.avg_pool2d(x, x.size()[2:])
            if self.att_gen == 'softmax':
                a = F.softmax(a, dim=1)
            elif self.att_gen == 'sigmoid':
                a = F.sigmoid(a)
                a = F.normalize(a, p=1, dim=1)
            else:
                raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
            x = x.view(b, t, -1)
            a = torch.unsqueeze(a, -1)
            a = a.expand(b, t, self.feat_dim)
            att_x = torch.mul(x, a)
            att_x = torch.sum(att_x, 1)

            f = att_x.view(b, self.feat_dim)
            if not self.training:
                return f
            y = self.classifier(f)

            if self.loss == {'xent'}:
                return y
            elif self.loss == {'xent', 'htri'}:
                return y, f, None
            elif self.loss == {'cent'}:
                return y, f, None
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))
        else:
            #con Depth
            b = x.size(0)
            t = x.size(1)
            bd = z.size(0)
            td = z.size(1)

            #Depth
            z = z.view(bd * td, z.size(2), z.size(3), z.size(4))
            z1 = self.base(z)
            ad = F.relu(self.attention_conv(z1))
            ad = ad.view(bd, td, self.middle_dim)
            ad1 = ad.permute(0, 2, 1)
            ad2 = F.relu(self.attention_tconv(ad1))
            ad2 = ad2.view(bd, td)
            z2= F.avg_pool2d(z1, z1.size()[2:])
            z2 = z2.view(b, t, -1)

            #RGB
            x = x.view(b * t, x.size(2), x.size(3), x.size(4))
            x = self.base(x)
            x = torch.add(x, z1)   # FUSIONE 1
            a = F.relu(self.attention_conv(x))
            a = a.view(b, t, self.middle_dim)
            a = a.permute(0, 2, 1)
            a = torch.add(a, ad1)  # FUSIONE 2
            a = F.relu(self.attention_tconv(a))
            a = a.view(b, t)
            a = torch.add(a, ad2)  # FUSIONE 3
            x = F.avg_pool2d(x, x.size()[2:])

            if self.att_gen == 'softmax':
                a = F.softmax(a, dim=1)
            elif self.att_gen == 'sigmoid':
                a = F.sigmoid(a)
                a = F.normalize(a, p=1, dim=1)
            else:
                raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
            x = x.view(b, t, -1)
            x = torch.add(x, z2)  # FUSIONE 4
            a = torch.unsqueeze(a, -1)
            a = a.expand(b, t, self.feat_dim)
            att_x = torch.mul(x, a)
            att_x = torch.sum(att_x, 1)

            f = att_x.view(b, self.feat_dim)
            if not self.training:
                return f
            y = self.classifier(f)

            if self.loss == {'xent'}:
                return y
            elif self.loss == {'xent', 'htri'}:
                return y, f, None
            elif self.loss == {'cent'}:
                return y, f, None
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50RNN(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50RNN, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.hidden_dim = 512
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
    def forward(self, x, z, use_depth=False):
        if not use_depth:
            #senza Depth - Codice Originale
            b = x.size(0)
            t = x.size(1)
            x = x.view(b*t,x.size(2), x.size(3), x.size(4))
            x = self.base(x)
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(b,t,-1)
            output, (h_n, c_n) = self.lstm(x)
            output = output.permute(0, 2, 1)
            f = F.avg_pool1d(output, t)
            f = f.view(b, self.hidden_dim)
            if not self.training:
                return f
            y = self.classifier(f)

            if self.loss == {'xent'}:
                return y
            elif self.loss == {'xent', 'htri'}:
                return y, f, None
            elif self.loss == {'cent'}:
                return y, f, None
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))

        else:
            #con Depth
            b = x.size(0)
            t = x.size(1)
            bd = z.size(0)
            td = z.size(1)

            #Depth
            z = z.view(bd * td, z.size(2), z.size(3), z.size(4))
            z1 = self.base(z)
            z2 = F.avg_pool2d(z1, z1.size()[2:])
            z2 = z2.view(bd, td, -1)
            output_d, (h_n, c_n) = self.lstm(z2)
            output_d = output_d.permute(0, 2, 1)
            fd = F.avg_pool1d(output_d, td)
            fd = fd.view(bd, self.hidden_dim)

            #RGB
            x = x.view(b * t, x.size(2), x.size(3), x.size(4))
            x = self.base(x)
            x = torch.add(x, z1)  # FUSIONE 1
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(b, t, -1)
            x = torch.add(x, z2)  # FUSIONE 2
            output, (h_n, c_n) = self.lstm(x)
            output = output.permute(0, 2, 1)
            output = torch.add(output, output_d)  # FUSIONE 3
            f = F.avg_pool1d(output, t)
            f = f.view(b, self.hidden_dim)
            f = torch.add(f, fd)  # FUSIONE 4
            if not self.training:
                return f
            y = self.classifier(f)

            if self.loss == {'xent'}:
                return y
            elif self.loss == {'xent', 'htri'}:
                return y, f, None
            elif self.loss == {'cent'}:
                return y, f, None
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))

'''
class ResNet50RNN_2(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50RNN, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.hidden_dim = 512
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.bilinear=nn.Bilinear(2048/4,2048/4,2048/4)
        self.lstm = nn.LSTM(input_size=self.feat_dim/4, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
    def forward(self, x, z):
        b = x.size(0)
        t = x.size(1)
        bd = z.size(0)
        bd = z.size(1)
        #RGB
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b,t,-1)
        output, (h_n, c_n) = self.lstm(x)
        output = output.permute(0, 2, 1)
        f = F.avg_pool1d(output, t)
        f = f.view(b, self.hidden_dim)
        #Depth
        z = z.view(bd*td,z.size(2), z.size(3), z.size(4))
        z = self.base(z)
        z = F.avg_pool2d(z, z.size()[2:])
        z = z.view(bd,td,-1)
        outputd, (h_n, c_n) = self.lstm(z)
        outputd = outputd.permute(0, 2, 1)
        fd = F.avg_pool1d(outputd, td)
        fd = fd.view(bd, self.hidden_dim)
        if not self.training:
            return f, fd

        y = self.bilinear(f,fd) #Uniamo
        #riaggiustiamo dimensioni
        f = f.view(16, -1)
        fd = fd.view(16, -1)
        y = y.view(16,-1)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f, fd
        elif self.loss == {'cent'}:
            return y, f, fd
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
'''
