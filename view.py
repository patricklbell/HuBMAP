from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import pyqtgraph as qtg

from functools import partial
import sys
import numpy as np
import pandas as pd
from PIL import Image

from utils.utils import coordinates_to_mask, mask_to_image


class MainWindow(QMainWindow):
    def __init__(self, tile_meta_filename="data/tile_meta.csv", wsi_meta_filename="data/wsi_meta.csv", labels_filename="data/polygons.jsonl",
                 colors={"blood_vessel": (100, 0, 0, 60), "glomerulus": (100, 100, 100, 40), "unsure": (100, 0, 100, 40)}):
        super().__init__()
        self.setWindowTitle("HuBMAP Viewer")

        self.colors = colors
        self.masks = {}

        self.tile_meta = pd.read_csv(tile_meta_filename)
        self.wsi_meta = pd.read_csv(wsi_meta_filename)
        self.labels = pd.read_json(labels_filename, lines=True)

        self.wsi_label = QLabel()
        self.tile_label = QLabel()
        self.mask_hbox = QHBoxLayout()

        self.tile_combo = QComboBox()
        self.tile_combo.addItems(self.tile_meta['id'].tolist())
        self.tile_combo.currentIndexChanged.connect(self.tileComboChanged)

        self.tile_meta.set_index(['id'], inplace=True)
        self.wsi_meta.set_index(['source_wsi'], inplace=True)
        self.labels.set_index(['id'], inplace=True)

        self.tile_view = qtg.ImageView()
        self.tile_view.show()

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.tile_combo)
        hbox.addWidget(self.tile_label, stretch=1)
        hbox.addLayout(self.mask_hbox)
        hbox.addWidget(self.wsi_label, alignment=Qt.AlignmentFlag.AlignRight)

        vbox.addLayout(hbox)
        vbox.addWidget(self.tile_view)

        vboxw = QWidget()
        vboxw.setLayout(vbox)
        self.setCentralWidget(vboxw)

        self.loadTile(self.tile_combo.currentText())

    def tileComboChanged(self, i):
        self.loadTile(self.tile_combo.currentText())

    def toggleMask(self, name):
        if self.masks[name]["checkbox"].isChecked():
            self.tile_view.getView().addItem(self.masks[name]["image"])
        else:
            self.tile_view.getView().removeItem(self.masks[name]["image"])

    def loadTile(self, id):
        data = np.array(Image.open(f"data/train/{id}.tif"))
        self.tile_view.setImage(data)

        tile_meta = self.tile_meta.loc[id]
        self.tile_label.setText(
            f"Dataset {tile_meta.dataset}, i:{tile_meta.i} j:{tile_meta.j}")

        for name, mask in self.masks.items():
            self.tile_view.getView().removeItem(mask["image"])
            self.mask_hbox.removeWidget(mask["checkbox"])
        self.masks = {}

        try:
            wsi_meta = self.wsi_meta.loc[tile_meta['source_wsi']]
            self.wsi_label.setText(
                f"Age {wsi_meta.age} Sex {wsi_meta.sex} Race {wsi_meta.race} Height {wsi_meta.height}cm Weight {wsi_meta.weight}kg BMI {wsi_meta.bmi}")
        except:
            self.wsi_label.setText(
                f"Error opening WSI metadata {tile_meta['source_wsi']}")

        try:
            label = self.labels.loc[id]
        except:
            return

        coordinates = {}
        for annotation in label['annotations']:
            type = annotation['type']
            coordinates[type] = (coordinates[type] if type in coordinates else [
            ]) + [np.array(x) for x in annotation['coordinates']]

        for type, coordinates in coordinates.items():
            checkbox = QCheckBox(text=type)
            self.mask_hbox.addWidget(checkbox)

            self.masks[type] = {
                "checkbox": checkbox,
                "image": qtg.ImageItem(mask_to_image(coordinates_to_mask(coordinates), self.colors[type]))
            }

            checkbox.toggled.connect(partial(self.toggleMask, type))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
