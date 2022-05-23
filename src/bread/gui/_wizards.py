from typing import Optional, List
from qtpy.QtWidgets import QWidget, QMenuBar, QMainWindow, QFormLayout, QVBoxLayout, QLabel, QHBoxLayout, QGridLayout, QPushButton, QCheckBox, QSlider, QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox, QWizard, QWizardPage
import bread.algo.lineage

class NameAndDescription(QWidget):
	def __init__(self, name: str, desc: str, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.label_name = QLabel(name, self)
		self.label_name.setWordWrap(True)
		self.label_desc = QLabel(desc, self)
		self.label_desc.setStyleSheet('font-size: 0.7em; color: #555; min-width: 300px;')
		self.label_desc.setWordWrap(True)

		self.setLayout(QVBoxLayout())
		self.layout().addWidget(self.label_name)
		self.layout().addWidget(self.label_desc)

		self.layout().setContentsMargins(0, 0, 0, 0)

class WizardPageBudneckParams(QWizardPage):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.nn_threshold = QDoubleSpinBox(self)
		self.nn_threshold.setMinimum(0)
		self.nn_threshold.setSingleStep(0.5)

		self.setLayout(QFormLayout())
		self.layout().addRow(
			NameAndDescription(
				'Nearest neighbour threshold',
				'cell masks separated by less than this threshold are considered neighbors, by default 8.0'
			),
			self.nn_threshold
		)

		self.registerField('nn_threshold', self.nn_threshold)

	def initializePage(self):
		self.nn_threshold.setValue(bread.algo.lineage._lineage.LineageGuesser.nn_threshold)

class WizardBudneck(QWizard):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.addPage(WizardPageBudneckParams(self))
		self.setWindowTitle('bread GUI - budneck guesser')