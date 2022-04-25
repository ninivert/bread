from optparse import Option
from cv2 import line
from qtpy import QtGui, QtWidgets, QtCore
from qtpy.QtWidgets import QWidget, QMenuBar, QMainWindow, QVBoxLayout, QLabel, QHBoxLayout, QGridLayout, QPushButton, QCheckBox, QSlider, QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox, QSpinBox, QFileDialog, QMessageBox
from qtpy.QtGui import QAction, QIcon
from qtpy.QtCore import QObject, Signal, Slot
import pyqtgraph as pg
from pathlib import Path
from typing import Optional, List
import numpy as np

from ._state import AppState
from ._utils import lerp
from bread.data import Lineage, Microscopy, Segmentation

__all__ = ['App', 'APP_STATE']

APP_STATE = AppState()

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


class Viewer(QWidget):
	class Layers(QWidget):
		# TODO : show opened file name
		# TODO : button to close the file
		# TODO : open file to start warning
		# TODO : time shape mismatch warning
		class Layer(QGroupBox):
			def __init__(self, parent: Optional[QWidget] = None, *args, **kwargs) -> None:
				super().__init__(parent, *args, **kwargs)

				self.openbtn = QPushButton('Open')
				self.openbtn.setIcon(QIcon('src/bread/gui/fugue-icons-3.5.6/icons-shadowless/folder-open-image.png'))
				self.opacityslider = QSlider(QtCore.Qt.Horizontal)
				self.opacityslider.setMinimum(0)
				self.opacityslider.setMaximum(10)
				self.opacityslider.setSingleStep(1)
				self.opacityslider.setValue(10)
				self.setLayout(QHBoxLayout())
				self.layout().addWidget(self.openbtn)
				self.layout().addWidget(self.opacityslider)

		def __init__(self, parent: Optional[QWidget] = None) -> None:
			super().__init__(parent)

			self.segmentation = Viewer.Layers.Layer(title='Segmentation')
			self.segmentation.opacityslider.valueChanged.connect(lambda val: APP_STATE.set_opacity_segmentation(lerp(val, self.segmentation.opacityslider.minimum(), self.segmentation.opacityslider.maximum(), 0, 1)))
			self.segmentation.openbtn.clicked.connect(self.file_open_segmentation)
			self.microscopy = Viewer.Layers.Layer(title='Microscopy')
			self.microscopy.opacityslider.valueChanged.connect(lambda val: APP_STATE.set_opacity_microscopy(lerp(val, self.microscopy.opacityslider.minimum(), self.microscopy.opacityslider.maximum(), 0, 1)))
			self.microscopy.openbtn.clicked.connect(self.file_open_microscopy)
			self.budneck = Viewer.Layers.Layer(title='Budneck')
			self.budneck.opacityslider.valueChanged.connect(lambda val: APP_STATE.set_opacity_budneck(lerp(val, self.budneck.opacityslider.minimum(), self.segmentation.opacityslider.maximum(), 0, 1)))
			self.budneck.openbtn.clicked.connect(self.file_open_budneck)
			self.nucleus = Viewer.Layers.Layer(title='Nucleus')
			self.nucleus.opacityslider.valueChanged.connect(lambda val: APP_STATE.set_opacity_nucleus(lerp(val, self.nucleus.opacityslider.minimum(), self.segmentation.opacityslider.maximum(), 0, 1)))
			self.nucleus.openbtn.clicked.connect(self.file_open_nucleus)
			self.setLayout(QVBoxLayout())
			self.layout().setAlignment(QtCore.Qt.AlignTop)
			self.layout().setContentsMargins(0, 0, 0, 0)
			self.layout().addWidget(self.segmentation)
			self.layout().addWidget(self.microscopy)
			self.layout().addWidget(self.budneck)
			self.layout().addWidget(self.nucleus)
			self.setFixedWidth(200)  # TODO

		@Slot()
		def file_open_segmentation(self):
			filepath, filefilter = QFileDialog.getOpenFileName(self, 'Open segmentation', './', 'Segmentation files (*.h5)')
			filepath = Path(filepath)
			if filepath.is_dir():
				return  # user did not select anything
			
			# TODO : proper file opening support in bread.data
			# something like Segmentation.from_filepath_autodetect
			if filepath.suffix in ['.h5', '.hd5']:
				segmentation = Segmentation.from_h5(filepath)
			else:
				raise RuntimeError(f'Unsupported extension : {filepath.suffix}')

			APP_STATE.set_segmentation_data(segmentation)

		@Slot()
		def file_open_microscopy(self):
			# yes, i am repeating myself, but function binding would be overkill and this is more explicit
			filepath, filefilter = QFileDialog.getOpenFileName(self, 'Open microscopy', './', 'Microscopy files (*.tiff, *.tif)')
			filepath = Path(filepath)
			if filepath.is_dir():
				return  # user did not select anything

			if filepath.suffix in ['.tif', '.tiff']:
				microscopy = Microscopy.from_tiff(filepath)
			else:
				raise RuntimeError(f'Unsupported extension : {filepath.suffix}')

			APP_STATE.set_microscopy_data(microscopy)

		@Slot()
		def file_open_budneck(self):
			# yes, i am repeating myself, but function binding would be overkill and this is more explicit
			filepath, filefilter = QFileDialog.getOpenFileName(self, 'Open budneck channel', './', 'Budneck microscopy files (*.tiff, *.tif)')
			filepath = Path(filepath)
			if filepath.is_dir():
				return  # user did not select anything

			if filepath.suffix in ['.tif', '.tiff']:
				budneck = Microscopy.from_tiff(filepath)
			else:
				raise RuntimeError(f'Unsupported extension : {filepath.suffix}')

			APP_STATE.set_budneck_data(budneck)

		@Slot()
		def file_open_nucleus(self):
			# yes, i am repeating myself, but function binding would be overkill and this is more explicit
			filepath, filefilter = QFileDialog.getOpenFileName(self, 'Open nucleus channel', './', 'Nucleus microscopy files (*.tiff, *.tif)')
			filepath = Path(filepath)
			if filepath.is_dir():
				return  # user did not select anything

			if filepath.suffix in ['.tif', '.tiff']:
				nucleus = Microscopy.from_tiff(filepath)
			else:
				raise RuntimeError(f'Unsupported extension : {filepath.suffix}')

			APP_STATE.set_nucleus_data(nucleus)

	class Controls(QWidget):
		class Timeline(QWidget):
			def __init__(self, parent: Optional[QWidget] = None) -> None:
				super().__init__(parent)

				self.prevbtn = QPushButton('Previous frame')
				self.prevbtn.setIcon(QIcon('src/bread/gui/fugue-icons-3.5.6/icons-shadowless/arrow-180.png'))
				self.prevbtn.clicked.connect(lambda: APP_STATE.set_frame_index(APP_STATE.values.frame_index-1))
				self.nextbtn = QPushButton('Next frame')
				self.nextbtn.setIcon(QIcon('src/bread/gui/fugue-icons-3.5.6/icons-shadowless/arrow.png'))
				self.nextbtn.clicked.connect(lambda: APP_STATE.set_frame_index(APP_STATE.values.frame_index+1))
				
				self.framespinbox = QSpinBox()
				self.framespinbox.setMinimum(0)
				self.framespinbox.setMaximum(0)
				self.framespinbox.valueChanged.connect(APP_STATE.set_frame_index)
				APP_STATE.update_frame_index.connect(self.framespinbox.setValue)
				APP_STATE.update_frames_max.connect(lambda x: self.framespinbox.setMaximum(x-1))
				
				self.timeslider = QSlider(QtCore.Qt.Horizontal)
				self.timeslider.setMinimum(0)
				self.timeslider.setMaximum(0)
				self.timeslider.valueChanged.connect(APP_STATE.set_frame_index)
				APP_STATE.update_frame_index.connect(self.timeslider.setValue)
				APP_STATE.update_frames_max.connect(lambda x: self.timeslider.setMaximum(x-1))

				self.setLayout(QHBoxLayout())
				self.layout().addWidget(self.timeslider)
				self.layout().addWidget(self.framespinbox)
				self.layout().addWidget(self.prevbtn)
				self.layout().addWidget(self.nextbtn)

		def __init__(self, parent: Optional[QWidget] = None) -> None:
			super().__init__(parent)

			self.showids = QCheckBox('Shows IDs')
			self.showids.stateChanged.connect(APP_STATE.set_show_ids)
			self.showlin = QCheckBox('Shows lineage relations (TODO)')
			self.showids.stateChanged.connect(APP_STATE.set_show_lineage_arrow)
			self.time = Viewer.Controls.Timeline()
			self.setLayout(QHBoxLayout())
			self.layout().addWidget(self.showids)
			self.layout().addWidget(self.showlin)
			self.layout().addWidget(self.time)

	class Canvas(QWidget):
		def __init__(self, parent: Optional[QWidget] = None) -> None:
			super().__init__(parent)

			self.view = pg.GraphicsView()
			self.vb = pg.ViewBox()
			self.vb.setAspectLocked()
			self.view.setCentralItem(self.vb)
	
			self.img_segmentation = pg.ImageItem()
			self.img_microscopy = pg.ImageItem()
			self.img_budneck = pg.ImageItem()
			self.img_nucleus = pg.ImageItem()
			self.img_segmentation.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus)
			self.img_microscopy.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus)
			self.img_budneck.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus)
			self.img_nucleus.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus)
			self.img_segmentation_colormap = pg.ColorMap((0, 1), ('#000', '#00F'))
			self.img_microscopy_colormap = pg.ColorMap((0, 1), ('#000', '#FFF'))
			self.img_budneck_colormap = pg.ColorMap((0, 1), ('#000', '#0F0'))
			self.img_nucleus_colormap = pg.ColorMap((0, 1), ('#000', '#F00'))
			self.img_segmentation.setColorMap(self.img_segmentation_colormap)
			self.img_microscopy.setColorMap(self.img_microscopy_colormap)
			self.img_budneck.setColorMap(self.img_budneck_colormap)
			self.img_nucleus.setColorMap(self.img_nucleus_colormap)

			self.hist_microscopy = pg.HistogramLUTWidget()
			self.hist_microscopy.setImageItem(self.img_microscopy)
			self.hist_microscopy.gradient.setColorMap(self.img_microscopy_colormap)
			self.hist_budneck = pg.HistogramLUTWidget()
			self.hist_budneck.setImageItem(self.img_budneck)
			self.hist_budneck.gradient.setColorMap(self.img_budneck_colormap)
			self.hist_nucleus = pg.HistogramLUTWidget()
			self.hist_nucleus.setImageItem(self.img_nucleus)
			self.hist_nucleus.gradient.setColorMap(self.img_nucleus_colormap)

			self.text_cellids: List[pg.TextItem] = []

			self.vb.addItem(self.img_microscopy)
			self.vb.addItem(self.img_budneck)
			self.vb.addItem(self.img_nucleus)
			self.vb.addItem(self.img_segmentation)

			self.setLayout(QGridLayout())
			self.layout().setSpacing(0)
			self.layout().addWidget(self.view, 0, 0)
			self.layout().addWidget(self.hist_microscopy, 0, 1)
			self.layout().addWidget(self.hist_budneck, 0, 2)
			self.layout().addWidget(self.hist_nucleus, 0, 3)

			APP_STATE.update_segmentation_data.connect(self.update_segmentation)
			APP_STATE.update_microscopy_data.connect(self.update_microscopy)
			APP_STATE.update_budneck_data.connect(self.update_budneck)
			APP_STATE.update_nucleus_data.connect(self.update_nucleus)
			APP_STATE.update_segmentation_data.connect(self.update_text_cellids)
			APP_STATE.update_segmentation_opacity.connect(lambda opacity: self.img_segmentation.setOpacity(opacity))
			APP_STATE.update_microscopy_opacity.connect(lambda opacity: self.img_microscopy.setOpacity(opacity))
			APP_STATE.update_budneck_opacity.connect(lambda opacity: self.img_budneck.setOpacity(opacity))
			APP_STATE.update_nucleus_opacity.connect(lambda opacity: self.img_nucleus.setOpacity(opacity))
			APP_STATE.update_show_ids.connect(self.update_text_cellids)
			APP_STATE.update_frame_index.connect(self.update_segmentation)
			APP_STATE.update_frame_index.connect(self.update_microscopy)
			APP_STATE.update_frame_index.connect(self.update_budneck)
			APP_STATE.update_frame_index.connect(self.update_nucleus)
			APP_STATE.update_frame_index.connect(self.update_text_cellids)
			APP_STATE.update_centered_cellid.connect(self.update_centered_cellid)

		@Slot()
		def update_segmentation(self):
			if APP_STATE.data.segmentation is None:
				return

			self.img_segmentation.setImage(APP_STATE.data.segmentation.data[APP_STATE.values.frame_index], levels=(0, 1))

		@Slot()
		def update_microscopy(self):
			if APP_STATE.data.microscopy is None:
				return

			self.img_microscopy.setImage(APP_STATE.data.microscopy.data[APP_STATE.values.frame_index])

		@Slot()
		def update_budneck(self):
			if APP_STATE.data.budneck is None:
				return

			self.img_budneck.setImage(APP_STATE.data.budneck.data[APP_STATE.values.frame_index])

		@Slot()
		def update_nucleus(self):
			if APP_STATE.data.nucleus is None:
				return

			self.img_nucleus.setImage(APP_STATE.data.nucleus.data[APP_STATE.values.frame_index])

		@Slot()
		def update_text_cellids(self):
			if APP_STATE.data.segmentation is None:
				return

			# if not showing ids, just update visibility and quit
			if not APP_STATE.values.show_ids:
				for textitem in self.text_cellids:
					textitem.setVisible(APP_STATE.values.show_ids)
				return

			cellids = APP_STATE.data.segmentation.cell_ids(APP_STATE.values.frame_index)
			cms = APP_STATE.data.segmentation.cms(APP_STATE.values.frame_index)

			# remove unused text items
			while len(self.text_cellids) > len(cellids):
				item = self.text_cellids.pop()
				self.vb.removeItem(item)

			# add new text items as needed
			while len(self.text_cellids) < len(cellids):
				self.text_cellids.append(pg.TextItem(fill='#FFF8', color='#000', anchor=(0.5, 0.5)))
				self.vb.addItem(self.text_cellids[-1])

			# update labels and positions
			for textitem, cellid, cm in zip(self.text_cellids, cellids, cms):
				textitem.setText(f'{cellid:d}')
				textitem.setPos(*cm)
				textitem.setVisible(APP_STATE.values.show_ids)

		@Slot(int, int)
		def update_centered_cellid(self, timeid, cellid):
			if APP_STATE.data.segmentation is None:
				return

			center = APP_STATE.data.segmentation.cms(timeid, [cellid])[0]
			size = 100
			rect = (center[0]-size/2, center[1]-size/2, size, size)
			self.vb.setRange(QtCore.QRectF(*rect))

	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.layers = Viewer.Layers()
		self.controls = Viewer.Controls()
		self.canvas = Viewer.Canvas()

		self.setLayout(QGridLayout())
		# self.layout().setSpacing(0)
		# self.layout().setContentsMargins(0, 0, 0, 0)
		self.layout().addWidget(self.layers, 0, 0)
		self.layout().addWidget(self.canvas, 0, 1)
		self.layout().addWidget(self.controls, 1, 0, 1, 0)


class Editor(QWidget):
	class EditorTab(QWidget):
		class RowControls(QWidget):
			def __init__(self, parent: Optional[QWidget] = None) -> None:
				super().__init__(parent)

				# https://forum.qt.io/topic/93621/add-buttons-in-tablewidget-s-row/8
				# TODO : controls
				self.cellitem = QTableWidgetItem()
				self.delbtn = QPushButton('Delete row')
				self.addbtn = QPushButton('Add row')
				self.moveupbtn = QPushButton('Move up')
				self.movedownbtn = QPushButton('Move down')
				self.assignid = QPushButton('Assign ID')
				
				self.setLayout(QHBoxLayout())
				self.layout().addWidget(self.movedownbtn)
				self.layout().addWidget(self.moveupbtn)
				self.layout().addWidget(self.addbtn)
				self.layout().addWidget(self.delbtn)
				self.layout().addWidget(self.assignid)

		def __init__(self, parent: Optional[QWidget] = None) -> None:
			super().__init__(parent)

			self.rowcontrols = Editor.EditorTab.RowControls()
			self.table = QTableWidget()
			self.table.setColumnCount(3)
			self.table.setHorizontalHeaderLabels(['Parent', 'Bud', 'Time'])
			self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
			self.table.cellChanged.connect(lambda *x: print('do validation', x))
			self.table.cellClicked.connect(self.handle_cell_clicked)

			self.filepath: Optional[Path] = None
			
			self.setLayout(QVBoxLayout())
			self.layout().addWidget(self.rowcontrols)
			self.layout().addWidget(self.table)

		def open_lineage(self, lineage: Lineage, filepath: Optional[Path] = None):
			self.filepath = filepath
			nrows = len(lineage.time_ids)

			self.table.clearContents()
			self.table.setRowCount(nrows)

			for irow, (parent_id, bud_id, time_id) in enumerate(zip(lineage.parent_ids, lineage.bud_ids, lineage.time_ids)):
				self.table.setItem(irow, 0, QTableWidgetItem('{:d}'.format(parent_id)))
				self.table.setItem(irow, 1, QTableWidgetItem('{:d}'.format(bud_id)))
				self.table.setItem(irow, 2, QTableWidgetItem('{:d}'.format(time_id)))

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
			contents = self.table.item(irow, icol).text()
			timeid = self.table.item(irow, 2).text()
			
			try:
				contents = int(contents)
				timeid = int(timeid)
			except ValueError as e:
				raise RuntimeError(f'cell at irow={irow}, icol={icol} (or icol=2) contains non-digit data : {e}')

			if icol == 0 or icol == 1:
				APP_STATE.set_frame_index(timeid)
				# parent id or bud id
				APP_STATE.set_centered_cellid(timeid, contents)

			elif icol == 2:
				# time index
				APP_STATE.set_frame_index(timeid)

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
		tab: Editor.EditorTab = self.editortabs.currentWidget()

		# no filename has been defined
		if tab.filepath is None:
			self.file_saveas()
		else:
			tab.export_lineage().save_csv(tab.filepath)

	@Slot()
	def file_saveas(self):
		tab: Editor.EditorTab = self.editortabs.currentWidget()
		lineage = tab.export_lineage()
		filepath, filefilter = QFileDialog.getSaveFileName(self, 'Save lineage', './', 'Lineage CSV files (*.csv)')
		filepath = Path(filepath)
		lineage.save_csv(filepath)
		# lineage.save_csv(filepath.with_name(filepath.stem + '.autosave.csv'))  # TODO : change this to override !

	@Slot()
	def file_close(self):
		tabindex: int = self.editortabs.currentIndex()
		tab: Editor.EditorTab = self.editortabs.currentWidget()
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
		editortab = Editor.EditorTab()
		editortab.open_lineage(lineage, filepath)
		self.editortabs.addTab(editortab, filepath.name if filepath is not None else 'Unsaved lineage (*)')  # takes ownership
		self.editortabs.setCurrentWidget(editortab)
