import matplotlib.pyplot as plt
import numpy as np


class Functions:
    def __init__(self, x, a, b, alpha):
        self.x = x
        self.y = self.x.copy()
        self.a = a
        self.b = b
        self.alpha = alpha

    @staticmethod
    def _reverse(f, x, y):
        x_2 = x * np.cos(f) - y * np.sin(f)
        y_2 = x * np.sin(f) + y * np.cos(f)
        return x_2.copy(), y_2.copy()

    def parabola(self):
        y = self.a * np.power(self.x, 2) + self.b * self.x + np.random.randint(-100, 100, 1)
        x, y = self._reverse(self.alpha, self.x, y)
        return x, y

    def hyperbola(self):   # При вызове этого метода лучше определять self.x как np.arange(-30, 31, 0.5)
        y = (self.a / self.x) + self.b
        x, y = self._reverse(self.alpha, self.x, y)
        return x, y

    def abs(self):
        y = self.a * np.abs(self.x) + self.b
        x, y = self._reverse(self.alpha, self.x, y)
        return x, y

    def sigmoid(self):
        y = self.a / (1 + np.power(np.e, -self.x)) + self.b
        x, y = self._reverse(self.alpha, self.x, y)
        return x, y

    def linear(self):
        y = self.a * self.x + self.b
        x, y = self._reverse(self.alpha, self.x, y)
        return x, y

    def sin(self):
        y = self.a * np.sin(self.x) + self.b      # Метод не использовался при генерации данных
        x, y = self._reverse(self.alpha, self.x, y)
        return x, y

    def cos(self):
        y = self.a * np.cos(self.x) + self.b
        x, y = self._reverse(self.alpha, self.x, y)
        return x, y

    def logarifm(self):
        y = self.a * np.log(self.x) + self.b
        x, y = self._reverse(self.alpha, self.x, y)
        return x, y


class DataGenerator:
    def __init__(self, a_len, b_len, alpha_len, random_comand="randn", alpha_initialize="base_circle"):
        np.random.seed(17)
        if random_comand == "randn":
            self.a = np.random.randn(a_len)
            self.b = np.random.randn(b_len)
        elif random_comand == "randint":
            self.a = np.random.randint(-100, 100, a_len)
            self.b = np.random.randint(-100, 100, b_len)
        else:
            print("message: random_comand should be 'randn' or 'randint'")
            raise ValueError

        if alpha_initialize == "randn":
            self.alpha = np.random.randn(alpha_len)
        elif alpha_initialize == "base_circle":
            self.alpha = np.array([0, 1/8, -1/8, 1/6, -1/6, 1/3, -1/3, 1/4, -1/4, 1/3, -1/3,
                                    5/8, -5/8, 1/2, -1/2, 5/8, -5/8, 2/3, -2/3, 3/4, -3/4,
                                    5/6, -5/6, 7/8, -7/8, 1]) * np.pi
        else:
            print("message: alpha_initialize should be 'randn' or 'base_circle'")
            raise ValueError

        self.x = np.arange(-20, 21, 0.5)
        self.y = self.x

    def fig_plot(self, function, ax=True, save=False, plot=True, verbose=100):       #Если save=True - то укажите свой PATH --> код ниже
        counter = 0
        var_x = self.x.copy()
        var_y = self.y.copy()
        for theta in self.alpha:
            for alpha in self.a:
                for beta in self.b:
                    func = Functions(x=var_x, a=alpha, b=beta, alpha=theta)

                    if function == 'parabola':
                        x, y = func.parabola()

                    elif function == 'sigmoid':
                        x, y = func.sigmoid()

                    elif function == 'abs':
                        x, y = func.abs()

                    elif function == 'hyperbola':
                        x, y = func.hyperbola()

                    elif function == 'linear':
                        x, y = func.linear()

                    elif function == 'sin':
                        x, y = func.sin()

                    elif function == 'cos':
                        x, y = func.cos()

                    elif function == 'log':
                        x, y = func.logarifm()


                    if (y != var_y).any():
                        fig = plt.figure(frameon=False)
                        ax = fig.add_axes([0, 0, 1, 1])
                        ax.axis('off')
                        plt.plot(x, y)

                        if ax:
                            plt.axvline(0, linestyle="--")
                            plt.axhline(0, linestyle="--")

                        if plot:
                            plt.show()
                        else:
                            plt.close()

                        if save:
                            counter += 1
                            PATH = "C:\Project_images" + "\\" + str(function) + '_' + str(counter) + ".png"

                            if counter % verbose == 0:
                                print(PATH)
                            fig.savefig(PATH)
