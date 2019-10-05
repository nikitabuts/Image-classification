import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class Net(nn.Module):
    def __init__(self, pic_size=42*64*4):
        super().__init__()
        self.fc1 = nn.Linear(pic_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        return F.log_softmax(x, dim=1)


class Learn:
    def __init__(self, epochs, train_loader, pic_size=42*64*4, test_loader=None):
        self.net = Net()
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.pic_size = pic_size

    def accuracy(self, loader):
        total = 0
        correct = 0
        with torch.no_grad():
            for data in loader:
                X_, y_ = data
                output = self.net(X_.view(-1, self.pic_size))
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y_[idx].long():
                        correct += 1
                    total += 1
        return round(correct / total, 3)

    def fit(self, use_best_model=True, print_flag=True):
        if self.test_loader != None:
            scores_frame = pd.DataFrame(columns=['Epoch', 'Train_Accuracy', 'Test_Accuracy'])
        else:
            scores_frame = pd.DataFrame(columns=['Epoch', 'Train_Accuracy'])
        for epoch in range(self.epochs):
            for data in self.train_loader:
                X, y = data
                self.net.zero_grad()
                output = self.net(X.view(-1, self.pic_size))
                loss = F.nll_loss(output, y.long())
                loss.backward()
                self.optimizer.step()
            train_accuracy = self.accuracy(self.train_loader)

            if self.test_loader != None:
                test_accuracy = self.accuracy(self.test_loader)
                scores_frame.loc[epoch] = ['Epoch ' + str(epoch + 1)] + [train_accuracy, test_accuracy]
            else:
                scores_frame.loc[epoch] = ['Epoch ' + str(epoch + 1)] + [train_accuracy]

            if use_best_model:
                PATH = "C:\\Torch\\torch_" + str(epoch + 1) + ".pt"    #Укажите свой PATH
                torch.save(self.net.state_dict(), PATH)

            if print_flag:
                print("EPOCH # {}:".format(epoch + 1))
                print("-----loss: {}".format(loss))
                print("-----train accuracy: {}".format(train_accuracy))
                if self.test_loader != None:
                    print("-----test accuracy: {}".format(test_accuracy))
                print('')

        return self.net, scores_frame

    def get_best_model(self, scores_frame):
        PATH = "C:\\Torch\\torch_" + str(scores_frame['Train_Accuracy'].argmax() + 1) + ".pt"
        model = Net()
        model.load_state_dict(torch.load(PATH), strict=False)
        model.eval()
        model.train()
        return model

    def accuracy_by_classes(self, model, loader, dictionary={0: 'hyperbola', 1: 'sigmoid', 2: 'abs', 3: 'linear', 4: 'parabola'}):
        classes = np.arange(len(dictionary))
        correct_mass = np.zeros(len(dictionary))
        all_mass = np.zeros(len(dictionary))
        class_dict = dict(zip(np.arange(len(dictionary)), dictionary.keys().tolist()))

        for data in loader:
            X, y = data
            output = model(X.view(-1, self.pic_size))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx].long():
                    correct_mass[int(y[idx].tolist())] += 1
                else:
                    class_dict[int(y[idx].tolist())][int(torch.argmax(i).tolist())] += 1
                all_mass[int(y[idx].tolist())] += 1

        print("Accuracy:")
        for class_ in classes:
            print("   {} accuracy: {}".format(dictionary[class_],
                                                np.round((correct_mass[class_] / all_mass[class_]), 3)))
        print("------------------------")
        print("False by classes")
        for class_ in classes:
            print("   {} accuracy: {}".format(dictionary[class_], class_dict[class_]))

