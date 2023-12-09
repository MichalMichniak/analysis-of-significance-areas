import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from typing import List, Tuple
import datetime
import pandas as pd
import torchmetrics.classification

class ResNet50_model_transfer:
    """
    ResNet50 architecture without cuda
    from transfer learning
    """
    def __init__(self) -> None:
        pass

    def load(self, n_labels, new = False, conv_layers_train = False) -> None:
        #self.cuda = torch.device('cuda')
        self.n_labels = n_labels
        self.conv_layers_train = conv_layers_train
        if(new):
            new_classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=2048, out_features=1000, bias=True),
                torch.nn.Sigmoid(),
                torch.nn.Linear(in_features=1000, out_features=n_labels, bias=True),
                torch.nn.Softmax()
            )
            self.resnet50 = torchvision.models.resnet50(pretrained=True, progress=True)
            self.resnet50.fc = new_classifier
            self.new = True
        else:
            self.resnet50 = torch.load("./models/resnet50_model.pth")
            self.new = False
        self.resnet50.cuda()
        lst_blocks = [self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool, self.resnet50.layer1,
                      self.resnet50.layer2, self.resnet50.layer3, self.resnet50.layer4, self.resnet50.avgpool]
        for block in lst_blocks:
            for param in block.parameters():
                param.requires_grad = conv_layers_train


    def forward_pass(self, x : torch.Tensor):
        return self.resnet50(x)
    
    def train(self,nr_epoch, dataloader : DataLoader, test_dataloader : DataLoader, learning_rate = 1e-3, weight_decay = 0.0)->Tuple[List[float],List[float],List[float],List[float]]:
        """
        param:
            nr_epoch : Uint - number of epochs
            dataloader : Dataloader - dataloader for training process
            x_test : np.ndarray - test data
            y_test : np.ndarray - class of test data (evaluation)
            learning_rate : float - learning rate for backpropagationsteps
        return:
            train_loss : List[float] - training loss
            test_loss : List[float] - test loss
            train_accuracy : List[float] - training accuracy
            test_accuracy : List[float] - test accuracy
        """
        ### if resnet50_model.csv dont exist make it:
        try:
            with open(f"./models_logs/resnet50_model.csv","r") as to_copy_file:
                for line in to_copy_file:
                    pass
                best_accuracy = float(line.split(sep=',')[3])

        except:
            with open(f"./models_logs/resnet50_model.csv","w") as to_copy_file:
                pass
            best_accuracy = 0.0
        
        path = "./models/resnet50_model.pth"
        #TODO: other metrics (f1)
        test_accuracy = []
        train_accuracy = []
        test_loss = []
        train_loss = []

        mean_loss_tr = 0.0
        accuracy_tr = 0.0
        if self.conv_layers_train:
            optimizer = torch.optim.Adam(self.resnet50.parameters(), lr = learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(self.resnet50.fc.parameters(), lr = learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()
        loss_func = torch.nn.CrossEntropyLoss()
        acc_metric = torchmetrics.classification.Accuracy(task="multiclass",num_classes=self.n_labels).cuda()
        for epoch in range(nr_epoch):
            self.resnet50.train()
            mean_loss_tr = 0.0
            accuracy_tr = 0.0
            for (x_batch,y_batch),tqdm_progress in zip(iter(dataloader),tqdm(range(len(dataloader)-1))):
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
                y_pred = self.forward_pass(x_batch)
                loss = loss_func(y_pred,y_batch)
                accuracy_tr += acc_metric(y_pred,torch.argmax(y_batch, dim=1))
                mean_loss_tr += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            mean_loss_tr = float(mean_loss_tr)/len(dataloader)
            train_loss.append(float(mean_loss_tr))
            accuracy_tr = float(accuracy_tr)/len(dataloader)
            train_accuracy.append(float(accuracy_tr))
            mean_loss = 0.0
            accuracy = 0.0
            self.resnet50.eval()
            with torch.no_grad():
                for x_batch,y_batch in iter(test_dataloader):
                    # x = torch.unsqueeze(torch.from_numpy(x).T,0)
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                    y_pred = self.forward_pass(x_batch)
                    loss = loss_func(y_pred,y_batch)
                    mean_loss += loss
                    accuracy += acc_metric(y_pred,torch.argmax(y_batch, dim=1))
                    # if torch.argmax(y_pred) == torch.argmax(torch.unsqueeze(torch.from_numpy(y.astype(float)),0).cuda()):
                    #     accuracy += 1
                mean_loss = float(mean_loss)/len(test_dataloader)
                test_loss.append(float(mean_loss))
                accuracy = float(accuracy)/len(test_dataloader)
                test_accuracy.append(float(accuracy))
            if best_accuracy <= accuracy:
                torch.save(self.resnet50, path)
                if(self.new):
                    with open(f"./models_logs/resnet50_model.csv","w") as file:
                        file.write(f"{train_loss[-1]},{test_loss[-1]},{train_accuracy[-1]},{test_accuracy[-1]},{epoch}\n")
                    self.new = False
                else:
                    with open(f"./models_logs/resnet50_model.csv","a") as file:
                        file.write(f"{train_loss[-1]},{test_loss[-1]},{train_accuracy[-1]},{test_accuracy[-1]},{epoch}\n")
                
            if(epoch%5== 0):
                date = f"{datetime.datetime.now()}"
                date = "_".join(date.split())
                date = "_".join(date.split(":"))
                date = "_".join(date.split("."))
                torch.save(self.resnet50, f"./models/resnet50_{date}.pth")
                with open(f"./models_logs/resnet50_{date}.csv","w") as file:
                    with open(f"./models_logs/resnet50_model.csv","r") as to_copy_file:
                        for line in to_copy_file.readline():
                            file.write(line)
                        file.write(f"{train_loss[-1]},{test_loss[-1]},{train_accuracy[-1]},{test_accuracy[-1]},{epoch}\n")
                
            print(f"EPOCH: {epoch+1}, TEST_LOSS{test_loss[-1]}, TEST_ACCURACY{test_accuracy[-1]}, TRAIN_LOSS{train_loss[-1]}, TRAIN_ACCURACY{train_accuracy[-1]}")
        return train_loss,test_loss,train_accuracy,test_accuracy
    