if __name__ == '__main__':
    import NetLearning
    import Preprocessing
    import DataCreator
    import torch



train, test = Preprocessing.DataPreparation(Preprocessing.GetData().do(), test_size=0).train_test_split()
net, scores = NetLearning.Learn(epochs=30, train_loader=train).fit()
best = net.get_best_model()
