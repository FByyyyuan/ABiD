import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim

torch.manual_seed(0)

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

class Classifier(nn.Module):
    def __init__(self,class_num):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(256, class_num)
    
    def set_lambda(self, lambd):
        self.lambd = lambd

    def grad_reverse(x, lambd=1.0):
        return GradReverse(lambd)(x)

    def forward(self,x,reverse=False):
        if reverse:
            x =  GradReverse(self.lambd)(x)
        x = self.fc(x)
        return x

class ABiD(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.mu = self.args.mu
        self.root = self.args.root
        domain_list = [c for c in  os.listdir(self.root) if os.path.isdir(os.path.join(self.root,c))]
        self.Msdomain_flag = True if len(domain_list) == 1 else False
        self.classnum = 5
        self.C1 = Classifier(class_num=self.classnum).to(self.args.device)
        self.C2 = Classifier(class_num=self.classnum).to(self.args.device)
        self.opt_c1 = optim.Adam(self.C1.parameters(),lr=0.001, weight_decay=0.0005)
        self.opt_c2 = optim.Adam(self.C2.parameters(),lr=0.001, weight_decay=0.0005)
        self.pred_adj = []
        super(ABiD, self).__init__()

    def pthload(self):
        if os.path.exists(self.args.weightpath):
            self.model.load_state_dict(torch.load(self.args.weightpath))
            print('successful load weight!')
        else:
            print('not successful load weight.')
        if os.path.exists(self.args.classifier1path):
            self.C1.load_state_dict(torch.load(self.args.classifier1path))
            print('successful load classifer1 weight!')
        else:
            print('not successful load classifer1 weight.')
        if os.path.exists(self.args.classifier2path):
            self.C2.load_state_dict(torch.load(self.args.classifier2path))
            print('successful load classifer2 weight!')
        else:
            print('not successful load classifer2 weight.')
    
    def compute_adjustment(self, loader, tro, args,test = False):
        label_freq = {}
        if test:
            loader = dict(sorted(loader.items()))
            label_freq_array = np.array(list(loader.values()))
            label_freq_array = label_freq_array / label_freq_array.sum()
            adjustments = np.log(label_freq_array ** tro + 1e-12)
            adjustments = torch.from_numpy(adjustments)
            adjustments = adjustments.to(args.device)
            return adjustments
        else:
            for _, _, target, _, _ in loader:
                target = target.to(args.device)
                for j in target:
                    key = int(j.item())
                    label_freq[key] = label_freq.get(key, 0) + 1
            label_freq = dict(sorted(label_freq.items()))
            label_freq_array = np.array(list(label_freq.values()))
            label_freq_array = label_freq_array / label_freq_array.sum()
            adjustments = np.log(label_freq_array ** tro + 1e-12)
            
            adjustments = torch.from_numpy(adjustments)
            adjustments = adjustments.to(args.device)
            return adjustments

    def discrepancy_cdd(self, output_t1, output_t2):
        output_t1 = output_t1.to(torch.double)
        output_t2 = output_t2.to(torch.double)
        mul = output_t1.transpose(0, 1).mm(output_t2)
        cdd_loss =(torch.sum(mul) - torch.trace(mul))/(20*output_t1.shape[0]*output_t1.shape[0])
        return cdd_loss

    def discrepancy_js(self, output_1, output_2, get_softmax=True):
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        log_mean_output = ((output_1 + output_2)/2)
        return (KLDivLoss(log_mean_output, output_1) + KLDivLoss(log_mean_output, output_2))/2
    
    def discrepancy(self, output_1, output_2):
        return torch.mean(torch.abs(F.softmax(output_1,dim = 1) - F.softmax(output_2,dim = 1)))
    
    def con_matrix(self, Confusion_Matrix, a, b):
        if len(a)== len(b):
            for i in range(0,len(a)-1):
                Confusion_Matrix[a[i],b[i]] += 1
        else:
            print("error")
        return Confusion_Matrix
    
    def weighted_matrix(self,N):
        weighted = np.zeros((N,N)) 
        for i in range(len(weighted)):
            for j in range(len(weighted)):
                weighted[i][j] = float(((i-j)**2)/(N-1)**2) 
        return weighted
    
    def kappa_QW(self,Confusion_Matrix):
        Confusion_Matrix = Confusion_Matrix.numpy()
        Confusion_Matrix = np.transpose(Confusion_Matrix)
        weighted = self.weighted_matrix(5)
        act_hist = np.sum(Confusion_Matrix, axis=1)
        pred_hist = np.sum(Confusion_Matrix, axis=0)
        E = np.outer(act_hist, pred_hist)/np.sum(Confusion_Matrix)
        num = np.sum(np.multiply(weighted, Confusion_Matrix))
        den = np.sum(np.multiply(weighted, E))
        quadratic_weighted_kappa = 1-np.divide(num,den)
        return quadratic_weighted_kappa*100
    
    def test(self,model,dataloader,test_len):
        model.eval()
        correct_1 = 0
        C_correct1 = []
        kappa_1 = 0
        Confusion_Matrix_1 = torch.zeros((self.classnum,self.classnum))
        with torch.no_grad():
            for images, target in dataloader:
                images, target = images.to(self.args.device), target.to(self.args.device)
                features = self.model(images)
                pred1 = self.C1(features)
                pred1 = pred1.data.max(1)[1]
                Confusion_Matrix_1 = self.con_matrix(Confusion_Matrix_1,pred1,target)
                correct_1 += pred1.eq(target.data.view_as(pred1)).cpu().sum()
            kappa_1 = self.kappa_QW(Confusion_Matrix_1)
            C_sum_1 = Confusion_Matrix_1.sum(axis= 0)
            for i in range(len(C_sum_1)):
                C_correct1.append(float(100*Confusion_Matrix_1[i,i]/C_sum_1[i]))
            C_average_1 =(np.array(C_correct1)).mean()
        return (100. *correct_1/test_len),C_average_1,kappa_1
    
    def train(self, train_loader,target_loader,target_test_loader,test_len,val_len): 
        self.pthload()
        a_step = 0
        self.pred_adj = torch.zeros(1,5)
        self.pred_adj = self.pred_adj.to(self.args.device)
        self.pred_adj = self.compute_adjustment(loader=train_loader, tro=0.5, args = self.args)

        for epoch_counter in range(self.args.epochs):
            dataloader_iterator = iter(target_loader)

            for image_anchor, image_positive, train_label, anchor_domain, pos_domain in tqdm(train_loader):
                try:
                    images_t,_ = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(target_loader)
                    images_t, _ = next(dataloader_iterator)

                images_t = images_t.to(self.args.device)
                images = torch.cat((image_anchor,image_positive), dim = 0)
                images = images.type(torch.FloatTensor)
                labels = torch.cat((train_label,train_label), dim = 0).view(1,2*self.args.batch_size)
                domains = torch.cat((anchor_domain,pos_domain), dim = 0).view(1,2*self.args.batch_size)
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                domains = domains.to(self.args.device)
                
                criterion_CE= nn.CrossEntropyLoss().to(self.args.device)

                if epoch_counter >= a_step:
                    for i in range(2):
                        self.optimizer.zero_grad()
                        self.opt_c1.zero_grad()
                        self.opt_c2.zero_grad()
                        features = self.model(images)
                        output_s1 = self.C1(features)
                        output_s2 = self.C2(features)
                        output_s1 = output_s1                               - self.mu * self.pred_adj
                        output_s2 = output_s2           + self.pred_adj     + self.mu * self.pred_adj
                        features_T = self.model(images_t)
                        out_T1 = self.C1(features_T)   
                        out_T2 = self.C2(features_T)   
                        loss_dis = criterion_CE(output_s1,labels[-1]) + criterion_CE(output_s2,labels[-1]) #- self.discrepancy(out_T1, out_T2)
                        loss_dis.backward()
                        self.opt_c1.step()
                        self.opt_c2.step()
                    ##Step 2:F
                    for i in range(2):  
                        self.optimizer.zero_grad()
                        self.opt_c1.zero_grad()
                        self.opt_c2.zero_grad()
                        features = self.model(images)
                        output_s1 = self.C1(features)
                        output_s2 = self.C2(features)
                        output_s1 = output_s1                                           - self.mu * self.pred_adj
                        output_s2 = output_s2                   + self.pred_adj         + self.mu * self.pred_adj
                        features_T = self.model(images_t)
                        out_T1 = self.C1(features_T)           
                        out_T2 = self.C2(features_T)            
                        loss_dis2 = criterion_CE(output_s1,labels[-1])/4 + criterion_CE(output_s2,labels[-1])/4 + self.discrepancy(out_T1, out_T2)/2
                        loss_dis2.backward()
                        self.optimizer.step()
                    ## Step 3: CE
                    self.optimizer.zero_grad()
                    self.opt_c1.zero_grad()
                    self.opt_c2.zero_grad()
                    features = self.model(images)
                    output_s1 = self.C1(features)
                    output_s2 = self.C2(features)
                    output_s2 = output_s2 + self.pred_adj
                    loss = criterion_CE(output_s1, labels[-1]) + criterion_CE(output_s2, labels[-1])
                    loss.backward()
                    self.optimizer.step()
                    self.opt_c1.step()
                    self.opt_c2.step()

            if epoch_counter >= a_step:
                self.test(self.model,target_test_loader,test_len)
                self.scheduler.step()   
