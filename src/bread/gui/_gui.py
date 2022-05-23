from qtpy import QtGui, QtWidgets, QtCore
from qtpy.QtWidgets import QWidget, QMenuBar, QMainWindow, QVBoxLayout, QLabel, QHBoxLayout, QGridLayout, QPushButton, QCheckBox, QSlider, QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox, QSpinBox, QFileDialog, QMessageBox
from qtpy.QtGui import QAction, QIcon
from qtpy.QtCore import QObject, Signal, Slot
from typing import Optional, List
from ._state import APP_STATE
from ._editor import Editor
from ._viewer import Viewer

__all__ = ['App']

class App(QMainWindow):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.viewer = Viewer()
		self.editor = Editor()

		self.menu_app = self.menuBar().addMenu('Application')
		self.menu_app.addAction(QAction('Documentation', self))
		self.menu_app.addAction(QAction('About', self))
		self.menu_app.addSeparator()
		self.menu_app.addAction(QAction('Quit', self))

		self.setCentralWidget(QWidget())
		self.centralWidget().setLayout(QHBoxLayout())
		self.centralWidget().layout().addWidget(self.viewer)
		self.centralWidget().layout().addWidget(self.editor)

		# self.setStyleSheet('border: 1px solid red;')

	def closeEvent(self, event: QtGui.QCloseEvent) -> None:
		APP_STATE.closing.emit()
		event.accept()