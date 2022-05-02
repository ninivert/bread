from qtpy import QtGui, QtWidgets, QtCore
from qtpy.QtWidgets import QWidget, QMenuBar, QMainWindow, QVBoxLayout, QLabel, QHBoxLayout, QGridLayout, QPushButton, QCheckBox, QSlider, QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox, QSpinBox, QFileDialog, QMessageBox
from qtpy.QtGui import QAction, QIcon
from qtpy.QtCore import QObject, Signal, Slot
from typing import Optional, List
from pathlib import Path
import warnings
import numpy as np
from bread.data import Lineage, Microscopy, Segmentation
from ._state import APP_STATE

__all__ = ['Editor']

class RowControls(QWidget):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		# IDEA : inline row controls
		# https://forum.qt.io/topic/93621/add-buttons-in-tablewidget-s-row/8
		self.delbtn = QPushButton('Delete row')
		self.delbtn.setIcon(QIcon('src/bread/gui/fugue-icons-3.5.6/icons-shadowless/table-delete-row.png'))
		self.addbtn = QPushButton('Add row')
		self.addbtn.setIcon(QIcon('src/bread/gui/fugue-icons-3.5.6/icons-shadowless/table-insert-row.png'))
		self.moveupbtn = QPushButton('Move up')
		self.moveupbtn.setIcon(QIcon('src/bread/gui/fugue-icons-3.5.6/icons-shadowless/arrow-090.png'))
		self.movedownbtn = QPushButton('Move down')
		self.movedownbtn.setIcon(QIcon('src/bread/gui/fugue-icons-3.5.6/icons-shadowless/arrow-270.png'))
		
		self.setLayout(QHBoxLayout())
		self.layout().addWidget(self.moveupbtn)
		self.layout().addWidget(self.movedownbtn)
		self.layout().addWidget(self.addbtn)
		self.layout().addWidget(self.delbtn)

		self.layout().setContentsMargins(0, 0, 0, 0)


class EditorTab(QWidget):
	COLOR_ERR_PARSE = QtGui.QColor(0, 0, 0, 128)
	COLOR_ERR_TIMEID = QtGui.QColor(255, 0, 0, 128)
	COLOR_ERR_CELLID = QtGui.QColor(255, 0, 0, 128)
	COLOR_WARN_CELLID = QtGui.QColor(255, 64, 64, 128)
	COLOR_SPECIAL_CELLID = QtGui.QColor(0, 64, 255, 128)

	update_dirty = Signal(bool)

	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.rowcontrols = RowControls()
		self.rowcontrols.delbtn.clicked.connect(self.del_row)
		self.rowcontrols.addbtn.clicked.connect(self.add_row)
		self.rowcontrols.moveupbtn.clicked.connect(self.moveup_row)
		self.rowcontrols.movedownbtn.clicked.connect(self.movedown_row)

		self.table = QTableWidget()
		self.table.setColumnCount(3)
		self.table.setHorizontalHeaderLabels(['Parent', 'Bud', 'Time'])
		self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
		self.table.cellChanged.connect(self.validate_cell)
		APP_STATE.update_segmentation_data.connect(self.validate_all)
		self.table.cellClicked.connect(self.handle_cell_clicked)
		self.table.verticalHeader().setVisible(False)
		self.table.setSortingEnabled(False)

		self.filepath: Optional[Path] = None
		self.dirty: bool = False
		self.table.cellChanged.connect(lambda: self.update_dirty.emit(True))
		# TODO : dirty stuff
		
		self.setLayout(QVBoxLayout())
		self.layout().addWidget(self.rowcontrols)
		self.layout().addWidget(self.table)

	@Slot()
	def del_row(self):
		self.table.blockSignals(True)  # block signals while we edit the table
		irow = self.table.currentRow()
		if irow == -1:
			# no row selected
			return

		self.table.removeRow(irow)		
		self.table.blockSignals(False)  # restore signals

	@Slot()
	def add_row(self):
		irow = self.table.currentRow() + 1
		self.table.insertRow(irow)

		self.table.setItem(irow, 0, QTableWidgetItem(''))
		self.table.setItem(irow, 1, QTableWidgetItem(''))
		self.table.setItem(irow, 2, QTableWidgetItem('{:d}'.format(APP_STATE.values.frame_index)))

	@Slot()
	def moveup_row(self):
		self.table.blockSignals(True)  # block signals while we edit the table
		irow = self.table.currentRow()
		icol0 = self.table.currentColumn()

		if irow == -1 or irow == 0:
			# no row selected or is the top row
			return

		# save current row and insert a blank
		row_items = [self.table.takeItem(irow, icol) for icol in range(self.table.columnCount())]
		
		# move above row into current row
		for icol in range(self.table.columnCount()):
			item = self.table.takeItem(irow-1, icol)
			self.table.setItem(irow, icol, item)
			# print(self.table.rowCount())

		# move the saved items into above row
		for icol, item in enumerate(row_items):
			self.table.setItem(irow-1, icol, item)

		self.table.setCurrentCell(irow-1, icol0)
		self.table.blockSignals(False)  # restore signals

	@Slot()
	def movedown_row(self):
		self.table.blockSignals(True)  # block signals while we edit the table
		irow = self.table.currentRow()
		
		if irow == -1 or irow == self.table.rowCount()-1:
			# no row selected or is the bottom row
			return

		self.table.setCurrentCell(irow+1, self.table.currentColumn())
		self.moveup_row()
		self.table.setCurrentCell(irow+1, self.table.currentColumn())
		self.table.blockSignals(False)  # restore signals

	@Slot()
	def validate_all(self):
		for irow in range(self.table.rowCount()):
			self.validate_cell(irow, 0)
			self.validate_cell(irow, 1)
			self.validate_cell(irow, 2)

	@Slot(int, int)
	def validate_cell(self, irow: int, icol: int):
		item = self.table.item(irow, icol)
		content = self.parse_cell(irow, icol)

		if content is None:
			# invalid number format
			item.setBackground(self.COLOR_ERR_PARSE)
			item.setToolTip('[ERROR] Non-integer value')
			return

		if icol == 0 or icol == 1:
			# validate cell id
			timeid = self.parse_cell(irow, 2)

			if timeid is None:
				item.setBackground(self.COLOR_WARN_CELLID)
				item.setToolTip('[WARNING] Could not validate cell because time id is invalid')
				return

			# see first if the cell has a special id
			is_special = True
			try:
				Lineage.SpecialParentIDs(content)
			except ValueError:
				is_special = False
				# cell id is not a special id
			
			if is_special:
				item.setBackground(self.COLOR_SPECIAL_CELLID)
				item.setToolTip(f'[INFO] Cell {content} is special ({Lineage.SpecialParentIDs(content).name})')
				return

			if not APP_STATE.data.valid_cellid(content, timeid):
				item.setBackground(self.COLOR_ERR_CELLID)
				item.setToolTip(f'[ERROR] Cell {content} does not exist at frame {timeid}')
				return


		if icol == 2:
			# validate time id
			if not APP_STATE.data.valid_frameid(content):
				item.setBackground(self.COLOR_ERR_TIMEID)
				item.setToolTip('[ERROR] Frame index out of range')
				return

		# reset color if there is no error
		item.setBackground(QtGui.QBrush())
		item.setToolTip('')

	def open_lineage(self, lineage: Lineage, filepath: Optional[Path] = None):
		self.filepath = filepath
		nrows = len(lineage.time_ids)

		self.table.blockSignals(True)  # block signals while we edit the table

		self.table.clearContents()
		self.table.setRowCount(nrows)

		for irow, (parent_id, bud_id, time_id) in enumerate(zip(lineage.parent_ids, lineage.bud_ids, lineage.time_ids)):
			self.table.setItem(irow, 0, QTableWidgetItem('{:d}'.format(parent_id)))
			self.table.setItem(irow, 1, QTableWidgetItem('{:d}'.format(bud_id)))
			self.table.setItem(irow, 2, QTableWidgetItem('{:d}'.format(time_id)))
			# manually validate once the entire row is loaded
			self.validate_cell(irow, 0)
			self.validate_cell(irow, 1)
			self.validate_cell(irow, 2)

		self.table.blockSignals(False)  # restore signals

	def export_lineage(self):
		N: int = self.table.rowCount()
		parent_ids, bud_ids, time_ids = np.zeros(N, dtype=int), np.zeros(N, dtype=int), np.zeros(N, dtype=int)
		
		for irow in range(N):
			parent_ids[irow] = int(self.table.item(irow, 0).text())
			bud_ids[irow] = int(self.table.item(irow, 1).text())
			time_ids[irow] = int(self.table.item(irow, 2).text())
		
		return Lineage(parent_ids, bud_ids, time_ids)

	@Slot(int, int)
	def handle_cell_clicked(self, irow: int, icol: int):
		def handle_col_time(irow, icol):
			timeid = self.parse_cell(irow, icol)
			if timeid is not None:
				APP_STATE.set_frame_index(timeid)

		def handle_col_cell(irow, icol):
			cellid = self.parse_cell(irow, icol)
			timeid = self.parse_cell(irow, 2)
			if timeid is not None:
				APP_STATE.set_frame_index(timeid)
			if cellid is not None and timeid is not None:
				APP_STATE.set_centered_cellid(timeid, cellid)
	
		if icol == 0 or icol == 1:
			# parent id or bud id
			handle_col_cell(irow, icol)
		elif icol == 2:
			# time index
			handle_col_time(irow, icol)

	def parse_cell(self, irow, icol) -> Optional[int]:
		contents = self.table.item(irow, icol).text()
		if contents == '':
			return None
		try:
			contents = int(contents)
		except ValueError as e:
			warnings.warn(f'cell at irow={irow}, icol={icol} contains non-digit data : {e}')
			return None
		return contents


class Editor(QWidget):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.menubar = QMenuBar()
		self.menu_file = self.menubar.addMenu('&File')
		file_open_action = QAction('Open lineage', self)
		file_open_action.triggered.connect(self.file_open)
		self.menu_file.addAction(file_open_action)
		self.menu_file.addSeparator()
		file_save_action = QAction('Save lineage', self)
		file_save_action.triggered.connect(self.file_save)
		self.menu_file.addAction(file_save_action)
		file_saveas_action = QAction('Save lineage as', self)
		file_saveas_action.triggered.connect(self.file_saveas)
		self.menu_file.addAction(file_saveas_action)
		self.menu_file.addSeparator()
		file_close_action = QAction('Close current lineage', self)
		file_close_action.triggered.connect(self.file_close)
		self.menu_file.addAction(file_close_action)
		self.menu_new = self.menubar.addMenu('&New')
		# TODO : implement this
		self.menu_new.addAction(QAction('Guess lineage using budneck', self))
		# TODO : implement this
		self.menu_new.addAction(QAction('Guess lineage using expansion speed', self))
		self.menu_new.addSeparator()
		new_lineage_prefilled_action = QAction('Create pre-filled lineage file', self)
		new_lineage_prefilled_action.triggered.connect(self.new_lineage_prefilled)
		self.menu_new.addAction(new_lineage_prefilled_action)
		new_lineage_empty = QAction('Create empty lineage file', self)
		new_lineage_empty.triggered.connect(self.new_lineage_empty)
		self.menu_new.addAction(new_lineage_empty)
		self.menu_vis = self.menubar.addMenu('&Visualize')
		# TODO : implement this
		self.menu_vis.addAction(QAction('Open graph view', self))

		self.editortabs = QTabWidget()
		APP_STATE.update_add_lineage_data.connect(self.add_lineage)

		self.setLayout(QVBoxLayout())
		self.layout().addWidget(self.menubar)
		self.layout().addWidget(self.editortabs)

	@Slot()
	def file_open(self):
		filepath, filefilter = QFileDialog.getOpenFileName(self, 'Open lineage', './', 'Lineage CSV files (*.csv)')
		filepath = Path(filepath)
		if filepath.is_dir():
			return  # user did not select anything
		lineage = Lineage.from_csv(filepath)
		APP_STATE.add_lineage_data(lineage, filepath)

	@Slot()
	def file_save(self):
		tab: EditorTab = self.editortabs.currentWidget()

		# no filename has been defined
		if tab.filepath is None:
			self.file_saveas()
		else:
			lineage = tab.export_lineage()
			lineage.save_csv(tab.filepath)
			APP_STATE.set_current_lineage_data(lineage)

	@Slot()
	def file_saveas(self):
		tab: EditorTab = self.editortabs.currentWidget()
		lineage = tab.export_lineage()
		filepath, filefilter = QFileDialog.getSaveFileName(self, 'Save lineage', './', 'Lineage CSV files (*.csv)')
		filepath = Path(filepath)
		if filepath.is_dir():
			return  # user did not select anything
		lineage.save_csv(filepath)
		APP_STATE.set_current_lineage_data(lineage)
		# lineage.save_csv(filepath.with_name(filepath.stem + '.autosave.csv'))  # TODO : change this to override !

	@Slot()
	def file_close(self):
		tabindex: int = self.editortabs.currentIndex()
		tab: EditorTab = self.editortabs.currentWidget()
		# tab.graceful_close()
		# TODO : confirm if not saved
		self.editortabs.removeTab(tabindex)

	@Slot()
	def new_lineage_empty(self):
		lineage = Lineage(np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))
		APP_STATE.add_lineage_data(lineage)

	@Slot()
	def new_lineage_prefilled(self):
		if APP_STATE.data.segmentation is None:
			QMessageBox.warning(self, 'bread GUI warning', 'No segmentation loaded, unable to prefill new lineage file.\nCreate an empty lineage file instead, or load a segmentation.')
			return

		lineage = APP_STATE.data.segmentation.find_buds()
		APP_STATE.add_lineage_data(lineage)
	
	@Slot(Lineage, Path)
	def add_lineage(self, lineage: Lineage, filepath: Optional[Path]):
		editortab = EditorTab()
		editortab.open_lineage(lineage, filepath)
		self.editortabs.addTab(editortab, filepath.name if filepath is not None else 'Unsaved lineage (*)')  # takes ownership
		self.editortabs.setCurrentWidget(editortab)
		APP_STATE.set_current_lineage_data(lineage)

	# @Slot()
	# def update_tab_filename()