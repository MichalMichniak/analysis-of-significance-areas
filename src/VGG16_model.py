import torch
import torchvision
from src.Bigearth import Dataloader
from tqdm import tqdm
from typing import List, Tuple
import datetime
import pandas as pd

class VGG16_model_transfer:
    """
    VGG16 architecture without cuda
    from transfer learning
    """
    def __init__(self) -> None:
        pass

    def load(self, n_labels, new = False, conv_layers_train = False) -> None:
        #self.cuda = torch.device('cuda')
        if(new):
            new_classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(p=0.5, inplace=False),
                torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(p=0.5, inplace=False),
                torch.nn.Linear(in_features=4096, out_features=n_labels, bias=True),
                torch.nn.Softmax()
            )
            self.vgg16 = torchvision.models.vgg16_bn(pretrained=True, progress=True)
            self.vgg16.classifier = new_classifier
            self.new = True
        else:
            self.vgg16 = torch.load("./models/vgg_model.pth")
            self.new = False
        self.vgg16.cuda()
        for param in self.vgg16.features.parameters():
            param.requires_grad = conv_layers_train


    def forward_pass(self, x : torch.Tensor):
        return self.vgg16(x.cuda())
    
    def train(self,nr_epoch, dataloader : Dataloader, learning_rate = 1e-3, weight_decay = 0.0)->Tuple[List[float],List[float],List[float],List[float]]:
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
        ### if vgg_model.csv dont exist make it:
        try:
            with open(f"./models_logs/vgg_model.csv","r") as to_copy_file:
                for line in to_copy_file:
                    pass
                best_accuracy = float(line.split(sep=',')[3])

        except:
            with open(f"./models_logs/vgg_model.csv","w") as to_copy_file:
                pass
            best_accuracy = 0.0
        
        path = "./models/vgg_model.pth"
        #TODO: other metrics (f1)
        test_accuracy = []
        train_accuracy = []
        test_loss = []
        train_loss = []

        mean_loss_tr = 0
        accuracy_tr = 0
        optimizer = torch.optim.Adam(self.vgg16.classifier.parameters(), momentum=0.1, lr = learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()
        loss_func = torch.nn.CrossEntropyLoss()
        for epoch in range(nr_epoch):
            self.vgg16.train()
            mean_loss_tr = 0
            accuracy_tr = 0
            for (n,(x_batch,y_batch)),tqdm_progress in zip(enumerate(dataloader),tqdm(range(len(dataloader)-1))):
                for x,y in zip(x_batch,y_batch):
                    x = torch.unsqueeze(torch.from_numpy(x).T,0)
                    y_pred = self.forward_pass(x)
                    loss = loss_func(y_pred,torch.unsqueeze(torch.from_numpy(y.astype(float)),0).cuda())
                    with torch.no_grad():
                        mean_loss_tr += loss
                        if torch.argmax(y_pred) == torch.argmax(torch.unsqueeze(torch.from_numpy(y.astype(float)),0).cuda()):
                            accuracy_tr += 1
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            mean_loss_tr = mean_loss_tr/len(dataloader)
            train_loss.append(float(mean_loss_tr))
            accuracy_tr = accuracy_tr/len(dataloader)
            train_accuracy.append(float(accuracy_tr))
            mean_loss = 0
            accuracy = 0
            self.vgg16.eval()
            with torch.no_grad():
                for no_test in range(dataloader.test_len()):
                    x,y = dataloader.get_test(no_test)
                    x = torch.unsqueeze(torch.from_numpy(x).T,0)
                    y_pred = self.forward_pass(x)
                    loss = loss_func(y_pred,torch.unsqueeze(torch.from_numpy(y.astype(float)),0).cuda())
                    mean_loss += loss
                    if torch.argmax(y_pred) == torch.argmax(torch.unsqueeze(torch.from_numpy(y.astype(float)),0).cuda()):
                        accuracy += 1
                mean_loss = mean_loss/len(dataloader)
                test_loss.append(float(mean_loss))
                accuracy = accuracy/len(dataloader)
                test_accuracy.append(float(accuracy))
            if best_accuracy <= accuracy:
                torch.save(self.vgg16, path)
                if(self.new):
                    with open(f"./models_logs/vgg_model.csv","w") as file:
                        file.write(f"{train_loss[-1]},{test_loss[-1]},{train_accuracy[-1]},{test_accuracy[-1]}\n")
                    self.new = False
                else:
                    with open(f"./models_logs/vgg_model.csv","a") as file:
                        file.write(f"{train_loss[-1]},{test_loss[-1]},{train_accuracy[-1]},{test_accuracy[-1]}\n")
            date = f"{datetime.datetime.now()}"
            date = "_".join(date.split())
            date = "_".join(date.split(":"))
            date = "_".join(date.split("."))
            torch.save(self.vgg16, f"./models/vgg_{date}.pth")
            with open(f"./models_logs/vgg_{date}.csv","w") as file:
                with open(f"./models_logs/vgg_model.csv","r") as to_copy_file:
                    for line in to_copy_file.readline():
                        file.write(line)
                    file.write(f"{train_loss[-1]},{test_loss[-1]},{train_accuracy[-1]},{test_accuracy[-1]}\n")
            
            print(f"EPOCH: {epoch+1}, TEST_LOSS{mean_loss}, TEST_ACCURACY{accuracy}, TRAIN_LOSS{mean_loss}, TTRAIN_ACCURACY{accuracy}")
        return train_loss,test_loss,train_accuracy,test_accuracy