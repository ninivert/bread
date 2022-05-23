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
		show_doc = QAction('Documentation', self)
		show_doc.triggered.connect(self.show_doc)
		self.menu_app.addAction(show_doc)
		show_about = QAction('About', self)
		show_about.triggered.connect(self.show_about)
		self.menu_app.addAction(show_about)
		self.menu_app.addSeparator()
		action_quit = QAction('Quit', self)
		action_quit.triggered.connect(self.quit)
		self.menu_app.addAction(action_quit)

		self.setCentralWidget(QWidget())
		self.centralWidget().setLayout(QHBoxLayout())
		self.centralWidget().layout().addWidget(self.viewer)
		self.centralWidget().layout().addWidget(self.editor)

		# self.setStyleSheet('border: 1px solid red;')

	def show_doc(self):
		QMessageBox.information(self, 'bread GUI - documentation', '<a href="https://github.com/ninivert/bread">https://github.com/ninivert/bread</a>')

	def show_about(self):
		QMessageBox.information(self, 'bread GUI - about', 'bread GUI about')  # TODO

	def quit(self):
		self.close()

	def closeEvent(self, event: QtGui.QCloseEvent) -> None:
		APP_STATE.closing.emit()
		event.accept()