from os.path import isfile, exists
from PIL import Image
import torch
import numpy as np
import torch.utils.data as data_utils


class ImagePreprocessing:
    def __init__(self, PATH, size=(64, 64), functions=['hyperbola', 'sigmoid', 'abs', 'linear', 'parabola']):
        if exists(PATH):
            self.PATH = PATH                            #Укажите свой PATH
            self.size = size
            self.functions = functions                   #Массив названий функций
        else:
            raise EOFError

    def load_image(self, PATH):               #Возвращает матрицу пикселей картинки
        img = Image.open(PATH)
        img.thumbnail(self.size)
        return np.asarray(img)

    def preprocessing(self, start=1, end=2600, open_flag=False):            #У меня выполнялась ~10 - 12 минут (15000 картинок)
        data = np.array([[]])
        dictionary = dict(zip(self.functions, np.arange(len(self.functions))))
        for function in self.functions:
            if open_flag:
                PATH = self.PATH + "\\" + function + '_' + str(start) + ".png"     #Посмотреть на первую картинку в новом типе функции
                if isfile(PATH):
                    Image.open(PATH).show()
            else:
                for i in range(start, end + 1):
                    PATH = self.PATH + "\\" + function + '_' + str(i) + ".png"
                    if isfile(PATH):
                        img = self.load_image(PATH)
                        img = img.reshape(1, -1)        #Создаем вектор размера [1, img.shape[0] * img.shape[1] * img.shape[2]] из матрицы пикселей
                        img = np.append(img, np.array(dictionary[function])).reshape(1, -1)         #Добавляем метку изображения(тип функции, к которому относится картинка)
                        if data.shape != (1, 0):
                            data = np.append(data, img, axis=0)
                        else:
                            data = img
        return data


class GetData(ImagePreprocessing):
    def __init__(self, csv_PATH):
        super().__init__()
        if exists(csv_PATH):
            self.csv_PATH = csv_PATH
        else:
            raise EOFError

    def do(self, is_data=True, not_data=False):
        if is_data:        #Если есть csv файл с результатом выполнения метода preprocessing класса ImageProcessing
            return np.loadtxt(self.csv_PATH, delimiter=',')          #Выполняется ~ 3.5 минуты
        elif not_data:
            return self.preprocessing(end=5100)
        else:
            raise ValueError


class DataPreparation:
    def __init__(self, data: np.array, test_size = 0.1, batch_size = 8000):
        if 0 <= test_size < 1:
            self.test_size = test_size
        else:
            raise ValueError
        self.data = data.copy()
        self.batch_size = batch_size

    def train_test_split(self):
        np.random.shuffle(self.data)
        length = int((1 - self.test_size) * self.data.shape[0])
        X_train = torch.Tensor((self.data[:length, :-1] - np.mean(self.data[:length, :-1])) / np.abs(np.std(self.data[:length, :-1])))
        y_train = torch.Tensor(self.data[:length, -1])
        train = data_utils.TensorDataset(X_train, y_train)
        train_loader = data_utils.DataLoader(train, batch_size=self.batch_size, shuffle=True)

        if self.test_size != 0:
            X_test = torch.Tensor(
                (self.data[length:self.data.shape[0], :-1] - np.mean(self.data[length:self.data.shape[0], :-1])) / np.abs(
                    np.std(self.data[length:self.data.shape[0], :-1])))
            y_test = torch.Tensor(self.data[length:self.data.shape[0], -1])
            test = data_utils.TensorDataset(X_test, y_test)
            test_loader = data_utils.DataLoader(train, batch_size=100, shuffle=True)
        else:
            test_loader = None

        return train_loader, test_loader