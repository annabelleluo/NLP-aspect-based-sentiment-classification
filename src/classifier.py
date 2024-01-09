from typing import List
import math
import copy
from sklearn import metrics
import torch
import torch.nn as nn
import numpy as np

random_seed = 42 # or any of your favorite number
torch.manual_seed(random_seed) 
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

from torch.utils.data import DataLoader,ConcatDataset
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from transformers import BertModel
from dataset import ABSADataset,Tokenizer
from model import LCF_BERT

class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """


    ############################################# comp

    def __init__(self):

        #Input features
        self.inputs_col = ['concat_bert_indices', 'concat_segments_indices',
                           'text_bert_indices', 'aspect_bert_indices']


    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """

        self.device=device

        #Max sequence length
        max_seq_len = 80

        #Chosen BERT model
        pretrained_bert_name = 'activebus/BERT-DK_rest' 


        #Define tokenizer
        self.tokenizer = Tokenizer(max_seq_len, pretrained_bert_name)

        #Define model
        bert = BertModel.from_pretrained(pretrained_bert_name)
        self.model = LCF_BERT(bert=bert, dropout=0.4, bert_dim=768, polarities_dim=3, max_seq_len=max_seq_len,
                              device=self.device, SRD=3, local_context_focus='cdm')

        self.model.to(self.device)

        #Define dataset with augmentations
        trainset_orig = ABSADataset(train_filename, self.tokenizer)
        trainset_aug = ABSADataset(train_filename, self.tokenizer,augment=True, swap=False)
        trainset = ConcatDataset([trainset_orig, trainset_aug])
        valset = ABSADataset(dev_filename, self.tokenizer)
        
        #Define batch size
        batch_size = 16
        
        #Create dataloaders
        train_data_loader =  DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        val_data_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False)

        self.reset_params(self.model, torch.nn.init.xavier_uniform_)

        #Hyperparameters
        epochs = 15
        lr = 5e-6
        weight_decay = 0.01
        patience = 5
        
        #Loss function & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)

        max_val_acc, max_val_f1, max_val_epoch= 0, 0, 0
        best_model = copy.deepcopy(self.model)
        
        #Training
        for epoch in range(epochs):
            print('>' * 100)
            print(f'epoch: {epoch}')
            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()
            for batch in train_data_loader:
                optimizer.zero_grad()

                inputs = [batch[col].to(self.device) for col in self.inputs_col]
                outputs = self.model(inputs)
                targets = batch['polarity'].to(self.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)

            train_acc = n_correct / n_total
            train_loss = loss_total / n_total
            print('> average loss: {:.4f}, train_acc: {:.4f}'.format(train_loss, train_acc))
            
            #Validation
            val_acc, val_f1 = self.evaluate_acc_f1(val_data_loader)
            print('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            
            #Save model with best validation accuracy
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = epoch
                best_model = copy.deepcopy(self.model)

            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

            if epoch - max_val_epoch >= patience:
                print('>> early stop.')
                break

        print("Best Validation Accuracy : {:.4f}".format(max_val_acc))
        print("Best F1-score : {:.4f}".format(max_val_f1))
        self.model = best_model


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        self.model.to(self.device)
        
        #Define dataset & dataloader
        dataset = ABSADataset(data_filename, self.tokenizer)
        test_data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
        
        #Switch model to evaluation mode
        self.model.eval()

        all_preds = torch.tensor([]).to(self.device)
        for batch in test_data_loader:
            inputs = [batch[col].to(self.device) for col in self.inputs_col]
            outputs = self.model(inputs)
            batch_pred = torch.argmax(outputs, -1)
            all_preds = torch.concat((all_preds, batch_pred))

        all_preds = all_preds.tolist()
        result = list(map(lambda x: str(int(x)).replace('1', 'neutral').replace('0', 'negative').replace('2', 'positive'), all_preds))
        return result


    def reset_params(self, model, initializer):
        '''
        Resets the parameters of model
        '''
        for child in model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def evaluate_acc_f1(self, data_loader):
        '''
        Returns the accuracy and F1-score for data provided through a dataloader
        '''
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        #Switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                t_inputs = [batch[col].to(self.device) for col in self.inputs_col]
                t_outputs = torch.argmax(self.model(t_inputs), -1)
                t_targets = batch['polarity'].to(self.device)
                
                n_correct += (t_outputs==t_targets).sum().item()
                n_total += len(t_outputs)

                t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1, 2], average='weighted')
        return acc, f1

