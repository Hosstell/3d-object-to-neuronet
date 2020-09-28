from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, qApp, QAction
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt


NEURONET_PATH = 'neuronet.net'


class Neuronet(nn.Module):
    def __init__(self):
        super(Neuronet, self).__init__()

        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.LeakyReLU()
        self.fc3 = nn.BatchNorm1d(8)

        self.fc4 = nn.Linear(8, 16)
        self.fc5 = nn.LeakyReLU()
        self.fc6 = nn.BatchNorm1d(16)

        self.fc7 = nn.Linear(16, 32)
        self.fc8 = nn.LeakyReLU()
        self.fc9 = nn.BatchNorm1d(32)

        self.fc10 = nn.Linear(32, 64)
        self.fc11 = nn.LeakyReLU()
        self.fc12 = nn.BatchNorm1d(64)

        self.fc13 = nn.Linear(64, 128)
        self.fc14 = nn.LeakyReLU()
        self.fc15 = nn.BatchNorm1d(128)

        self.fc16 = nn.Linear(128, 10000)
        self.fc17 = nn.ReLU()


    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        x = self.fc10(x)
        x = self.fc11(x)
        x = self.fc12(x)
        x = self.fc13(x)
        x = self.fc14(x)
        x = self.fc15(x)
        x = self.fc16(x)
        x = self.fc17(x)

        return x


# Наследуемся от QMainWindow
class MainWindow(QMainWindow):
    # Переопределяем конструктор класса
    def __init__(self):
        # Обязательно нужно вызвать метод супер класса
        QMainWindow.__init__(self)

        # Нейросеть
        self.net = Neuronet()
        self.net.load_state_dict(torch.load(NEURONET_PATH))
        self.net.eval()

        self.setMinimumSize(QSize(480, 320))
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        grid_layout = QGridLayout(self)
        central_widget.setLayout(grid_layout)

        self.sld_1 = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sld_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)

        self.sld_1.valueChanged[int].connect(self.change_value_1)
        self.sld_2.valueChanged[int].connect(self.change_value_2)

        grid_layout.addWidget(self.sld_1, 1, 0)
        grid_layout.addWidget(self.sld_2, 2, 0)


        self.label = QLabel(self)
        self.label.setPixmap(self.generate_image(0.5, 0))
        self.label.resize(500, 500)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        grid_layout.addWidget(self.label, 0, 0)


    def generate_image(self, x, y):
        input = [x, y]
        input = torch.Tensor(input)
        input = input.view(1, 2)

        output = self.net(input)


        array = output.view(100, 100).detach().numpy()
        array = np.array(array, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image = new_image.resize((150, 150), Image.ANTIALIAS)

        qim = ImageQt(new_image)
        pix = QPixmap.fromImage(qim)
        return pix

    def change_value_1(self, value):
        value_1 = value / 100
        value_2 = self.sld_2.value() / 100
        self.label.setPixmap(self.generate_image(value_1, value_2))

    def change_value_2(self, value):
        value_1 = self.sld_1.value() / 100
        value_2 = value / 100
        self.label.setPixmap(self.generate_image(value_1, value_2))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())