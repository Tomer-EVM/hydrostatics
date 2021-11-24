import base64
import datetime
import io
import logging
import os.path
import sys
from queue import Queue
from time import sleep
from typing import List

import markdown
import numpy as np
import PyQt5
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PIL import Image
from PyQt5 import Qt, QtCore, uic
from PyQt5.QtWidgets import (
    QComboBox,
    QErrorMessage,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from hydrostatics.analysis import analyse_heel, analyse_trim, grid
import pandas as pd
from hydrostatics.mesh_processing import Mesh, close_ends, mirror_uv, save_uv
from hydrostatics.models import BuoyancyModel, load_hydro
from hydrostatics.optimize import solvers

logging.basicConfig(level="DEBUG")

if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

Ui_MainWindow, _ = uic.loadUiType(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "gui_src", "design.ui"))
)


class WorkerSignals(Qt.QObject):
    """Contains signals that can be emitted by Worker"""

    finished = Qt.pyqtSignal()
    progress = Qt.pyqtSignal(str)
    result = Qt.pyqtSignal(object)
    error = Qt.pyqtSignal(tuple)


class Worker(Qt.QRunnable):
    """A remote threaded worker for background calculations

    Attributes
    ----------
    f : function
        The function to execute
    args : list
        The arguments to pass to the function
    kwargs : dict
        The kwargs to pass to the function
    signals : Qt.QObject
        A class of signals that can be emitted
    """

    def __init__(self, f, *args, **kwargs):
        super(Worker, self).__init__()
        self.f = f
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Qt.pyqtSlot()
    def run(self):
        """Executes the function"""
        try:
            print("Starting")
            result = self.f(*self.args, **self.kwargs)
        except Exception as e:
            print(f"Errored: {e}")
            pass
        else:
            self.signals.result.emit(result)
        finally:
            print("Finished")
            self.signals.finished.emit()


class MainWindow(Qt.QMainWindow):
    """The main container for GUI logic

    Attributes
    ----------
    hydro : BuoyancyModel
        The model for storing meshes and doing calculations
    ui
        The class containing all layout and widget descriptions. Produced by QtDesigner and pyuic5
    thread_pool : Qt.QThreadPool
        For multithreading
    vtk_widget : QtInteractor
        The opengl context for use by pyvista

    """

    def __init__(self, show=True, hydro=None):
        super(MainWindow, self).__init__()

        if hydro is None:
            logging.info("Loading Model")
            hydro = BuoyancyModel()
        self.hydro = hydro

        self.initUI()
        self.setupVtkWindow()
        self.setupCallbacks()

        self.reset_globals()

        self.thread_pool = Qt.QThreadPool()

        if show:
            self.show()

    def initUI(self):
        logging.info("Init UI")
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Hydrostatics")
        self.setWindowIcon(
            Qt.QIcon(
                os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "gui_src", "logo.png")
                )
            )
        )
        self.setup_analysis()

    def setup_analysis(self):
        g: QWidget = self.ui.grid_analysis
        layout = QVBoxLayout(g)
        self.figure = Figure()
        self.figure.tight_layout()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        buttons = QWidget(g)
        QHBoxLayout(buttons)
        self.analysis_resolution = QSpinBox()
        self.analysis_resolution.setMinimum(2)
        self.analysis_resolution.setMaximum(1000)
        self.analysis_column = QComboBox()
        heel = QPushButton("Heel")
        trim = QPushButton("Trim")
        grid = QPushButton("Grid")
        download = QPushButton("CSV")
        buttons.layout().addWidget(self.analysis_resolution)
        buttons.layout().addWidget(self.analysis_column)
        buttons.layout().addWidget(heel)
        buttons.layout().addWidget(trim)
        buttons.layout().addWidget(grid)
        buttons.layout().addWidget(download)
        heel.clicked.connect(self.analysis_heel)
        trim.clicked.connect(self.analysis_trim)
        grid.clicked.connect(self.analysis_grid)
        download.clicked.connect(self.csv)
        self.analysis_column.currentIndexChanged.connect(self.plot_analysis)

        layout.addWidget(buttons)
        self._analysis_data = None
        self.x_data = None

    def analysis_trim(self):
        self._analysis_data = None
        self.w = Worker(
            analyse_trim,
            self.hydro,
            (self.ui.trimLB.value(), self.ui.trimUB.value()),
            self.analysis_resolution.value(),
        )
        self.w.signals.result.connect(self.finished_analysis)
        self.thread_pool.start(self.w)
        self.x_data = "trim"

    def analysis_heel(self):
        self._analysis_data = None
        self.w = Worker(
            analyse_heel,
            self.hydro,
            (self.ui.heelLB.value(), self.ui.heelUB.value()),
            self.analysis_resolution.value(),
        )
        self.w.signals.result.connect(self.finished_analysis)
        self.thread_pool.start(self.w)
        self.x_data = "heel"

    def analysis_grid(self):
        self._analysis_data = None
        self.w = Worker(
            grid,
            self.hydro,
            (self.ui.heelLB.value(), self.ui.heelUB.value()),
            (self.ui.trimLB.value(), self.ui.trimUB.value()),
            self.analysis_resolution.value(),
        )
        self.w.signals.result.connect(self.finished_analysis)
        self.thread_pool.start(self.w)
        self.x_data = ("heel", "trim")

    def csv(self):
        logging.info("saving csv data")
        if self._analysis_data is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage("Run an analysis before saving results")
            error_dialog.show()
            error_dialog.exec_()
        else:
            name = Qt.QFileDialog.getSaveFileName(self, "Save CSV", filter="(*.csv)")
            if name[0]:
                with open(name[0], "w") as f:
                    output = self._analysis_data.to_csv()
                    f.write(output)

    def finished_analysis(self, results: List[dict]):
        names = list(results[0].keys())
        data = {n: [r[n] for r in results] for n in names}
        self._analysis_data = pd.DataFrame(data=data)
        self.analysis_column.clear()
        self.analysis_column.addItems(self._analysis_data.columns.values.tolist())
        self.plot_analysis()

    def plot_analysis(self):
        if (
            self._analysis_data is not None
            and self.analysis_column.currentText() != ""
            and self.x_data is not None
        ):
            self.figure.clear()
            if isinstance(self.x_data, str):
                x = self._analysis_data[self.x_data]
                y = self._analysis_data[self.analysis_column.currentText()]
                ax = self.figure.add_subplot(111)
                ax.plot(x, y)
                ax.set_xlabel(self.x_data)
                ax.set_ylabel(self.analysis_column.currentText())
                self.canvas.draw()
            else:
                x = self._analysis_data[self.x_data[0]]
                y = self._analysis_data[self.x_data[1]]
                z = self._analysis_data[self.analysis_column.currentText()]
                ax = self.figure.add_subplot(projection="3d")
                ax.set_xlabel(self.x_data[0])
                ax.set_ylabel(self.x_data[1])
                ax.set_zlabel(self.analysis_column.currentText())
                ax.scatter(x, y, z)
                self.canvas.draw()

    def setupVtkWindow(self):
        logging.info("Init VTK")
        sleep(1)
        self.vtk_widget = QtInteractor(self)
        logging.info("Adding VTK widget")
        self.ui.viewerLayout.addWidget(self.vtk_widget.interactor)
        logging.info("Adding VTK callback")
        self.ui.buttonRefresh.clicked.connect(self.update_window)
        self.vtk_widget.set_background("grey")

    def setupCallbacks(self):
        logging.info("Creating callbacks")
        self.setupToolbar()
        self.setupMeshCallbacks()
        self.setupWeightCallbacks()
        self.setupSolverCallbacks()
        self.setupPositionCallbacks()
        self.setupResultsCallbacks()

    def setupResultsCallbacks(self):
        self.ui.saveResults.clicked.connect(self.print)
        self.ui.fullResults.clicked.connect(self.report)

    def setupToolbar(self):
        self.ui.actionSave_Project.triggered.connect(self.project_save)
        self.ui.actionOpen_Project.triggered.connect(self.project_open)
        self.ui.actionNew_Project.triggered.connect(self.new_project)
        self.ui.actionView_Documentation.triggered.connect(self.view_documentation)

    def setupMeshCallbacks(self):
        self.ui.buttonLoadMesh.clicked.connect(self.mesh_open)
        self.ui.buttonRemoveMesh.clicked.connect(self.mesh_remove)
        self.ui.buttonSaveMesh.clicked.connect(self.mesh_save)
        self.ui.buttonSaveSubmerged.clicked.connect(self.mesh_save_submerged)
        self.ui.selectMesh.currentTextChanged.connect(self.mesh_select)

        self.ui.meshName.editingFinished.connect(self.mesh_rename)

        self.ui.meshXPos.valueChanged.connect(self.mesh_x_pos_change)
        self.ui.meshYPos.valueChanged.connect(self.mesh_y_pos_change)
        self.ui.meshZPos.valueChanged.connect(self.mesh_z_pos_change)
        self.ui.meshXRot.valueChanged.connect(self.mesh_x_rot_change)
        self.ui.meshYRot.valueChanged.connect(self.mesh_y_rot_change)
        self.ui.meshZRot.valueChanged.connect(self.mesh_z_rot_change)

        self.ui.showMesh.clicked.connect(self.mesh_shown)
        self.ui.activeMesh.clicked.connect(self.mesh_active)
        self.reset_meshes()

    def setupWeightCallbacks(self):
        self.ui.buttonAddWeight.clicked.connect(self.add_weight)
        self.ui.buttonRemoveWeight.clicked.connect(self.weight_remove)
        self.ui.selectWeight.currentIndexChanged.connect(self.weight_select)

        self.ui.weightName.editingFinished.connect(self.weight_rename)

        self.ui.weightXPos.valueChanged.connect(self.weight_x_pos_change)
        self.ui.weightYPos.valueChanged.connect(self.weight_y_pos_change)
        self.ui.weightZPos.valueChanged.connect(self.weight_z_pos_change)
        self.ui.weightMagnitude.valueChanged.connect(self.weight_magnitude_change)

        self.ui.showWeight.clicked.connect(self.weight_shown)
        self.ui.activeWeight.clicked.connect(self.weight_active)
        self.reset_weights()

    def setupSolverCallbacks(self):
        self.ui.buttonOptimize.clicked.connect(self.optimize)
        self.ui.selectOptimizer.insertItems(0, (s.__name__ for s in solvers))

    def setupPositionCallbacks(self):
        self.ui.buttonRecalculate.clicked.connect(self.recalculate)

    def new_project(self):
        """Replaces current BuoyancyModel with an empty BuoyancyModel, and clears most other values"""
        self.hydro = BuoyancyModel()
        self.reset_meshes()
        self.reset_weights()
        self.reset_globals()

    def project_open(self):
        """Opens a previously saved BuoyancyModel"""
        name = Qt.QFileDialog.getOpenFileName(self, "Open Project", filter="(*.hydro)")
        if name[0]:
            self.hydro = load_hydro(name[0])
            self.ui.selectMesh.clear()
            self.ui.selectMesh.addItems(self.hydro.meshes.keys())
            self.reset_weights()
            self.reset_globals()

    def project_save(self):
        """Saves the BuoyancyModel to a file"""
        name = Qt.QFileDialog.getSaveFileName(self, "Save Project", filter="(*.hydro)")
        if name[0]:
            self.hydro.save(name[0])

    def view_documentation(self):
        """Opens the documentation in the results view"""
        file_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "docs",
                "build",
                "html",
                "Usage",
                "gui_help.html",
            )
        )
        if os.path.isfile(file_path):
            self.ui.resultsOutput.load(Qt.QUrl.fromLocalFile(file_path))
            self.ui.resultsOutput.show()
        else:
            self.ui.resultsOutput.setHtml(
                r"<h1>Docs not built. Run `make html` from docs directory"
            )

    def mesh_open(self):
        """Loads a mesh file into the buoyancy model

        Supports .dxf files, as well as all mesh formats supported by trimesh
        """
        name = Qt.QFileDialog.getOpenFileName(
            self,
            "Open Mesh",
            filter="All Files (*.*) ;; DXF (*.dxf) ;; Numpy mesh (*.npy) ;; Trimesh (*.stl *.stl_ascii *.dict *.dict64 *.json *.msgpack *.ply *.obj *.off *.glb *.gltf *.xyz *.zip *.tar.bz2 *.tar.gz)",
        )
        if name:
            if os.path.splitext(name[0])[1] == ".dxf":
                dialog = LoadPopup()
                if dialog.exec():
                    dim1, dim2, mirror, close = dialog.getInputs()
                    shape = (dim1, dim2)
                else:
                    shape = None
                    mirror = True
                    close = True
            else:
                shape = None
                mirror = False
                close = False
            new = self.hydro.load_mesh(name[0], shape=shape)
            for mesh in new:
                if mirror:
                    self.hydro.mirror(mesh)
                if close:
                    self.hydro.close_ends(mesh)
            self.reset_meshes()
            if name:
                self.ui.selectMesh.setCurrentText(name[-1])

    def mesh_remove(self):

        name = self.ui.selectMesh.currentText()
        if name:
            self.hydro.meshes.pop(name)
            self.hydro.transformed.pop(name)
            self.hydro.active_mesh.pop(name)
            self.hydro.show_mesh.pop(name)
            self.hydro.local_position.pop(name)
            self.hydro.local_rotation.pop(name)
            self.hydro.faces_below_water.pop(name)
            self.hydro.water_edge.pop(name)
            self.reset_meshes()
            self.hydro.run()

    def mesh_save(self):
        """Saves the currently selected mesh to a file

        File format can be selected from range available from trimesh
        """
        name = self.ui.selectMesh.currentText()
        if name:
            rotation = (0, 0, 0)
            if self.ui.earthCoordinates.isChecked():
                rotation = (self.hydro.heel, self.hydro.trim, self.hydro.leeway)
            file = Qt.QFileDialog.getSaveFileName(
                self,
                "Save Mesh",
                filter="(*.stl) ;; (*.off) ;; (*.ply) ;; (*.dae) ;; (*.json) ;; (*.dict) ;; (*.glb) ;; (*.dict64) ;; (*.msgpack)",
            )
            self.hydro.transformed[name].save(file[0], rotation)

    def mesh_save_submerged(self):
        """Saves the submerged component of the selected mesh to a file

        File format can be selected from range available from trimesh
        """
        name = self.ui.selectMesh.currentText()
        if name:
            file = Qt.QFileDialog.getSaveFileName(
                self,
                "Save Mesh",
                filter="(*.stl) ;; (*.off) ;; (*.ply) ;; (*.dae) ;; (*.json) ;; (*.dict) ;; (*.glb) ;; (*.dict64) ;; (*.msgpack)",
            )
            rotation = (0, 0, 0)
            if self.ui.earthCoordinates.isChecked():
                rotation = (self.hydro.heel, self.hydro.trim, self.hydro.leeway)
            self.hydro.faces_below_water[name].save(file[0], rotation=rotation)

    def mesh_rename(self):
        """Renames the currently selected mesh"""
        old = self.ui.selectMesh.currentText()
        if old:
            new = self.ui.meshName.text()
            self.hydro.rename_mesh(old, new)
            self.reset_meshes()
            self.ui.selectMesh.setCurrentText(new)

    def mesh_x_pos_change(self):
        """When the x position changes in the ui, update the stored value"""
        name = self.ui.selectMesh.currentText()
        if name:
            self.hydro.local_position[name][0] = self.ui.meshXPos.value()
            self.hydro.calculate_transformation()

    def mesh_y_pos_change(self):
        """When the y position changes in the ui, update the stored value"""
        name = self.ui.selectMesh.currentText()
        if name:
            self.hydro.local_position[name][1] = self.ui.meshYPos.value()
            self.hydro.calculate_transformation()

    def mesh_z_pos_change(self):
        """When the z position changes in the ui, update the stored value"""
        name = self.ui.selectMesh.currentText()
        if name:
            self.hydro.local_position[name][2] = self.ui.meshZPos.value()
            self.hydro.calculate_transformation()

    def mesh_x_rot_change(self):
        """When the x rotation changes in the ui, update the stored value"""
        name = self.ui.selectMesh.currentText()
        if name:
            self.hydro.local_rotation[name][0] = self.ui.meshXRot.value()
            self.hydro.calculate_transformation()

    def mesh_y_rot_change(self):
        """When the y rotation changes in the ui, update the stored value"""
        name = self.ui.selectMesh.currentText()
        if name:
            self.hydro.local_rotation[name][1] = self.ui.meshYRot.value()
            self.hydro.calculate_transformation()

    def mesh_z_rot_change(self):
        """When the z rotation changes in the ui, update the stored value"""
        name = self.ui.selectMesh.currentText()
        if name:
            self.hydro.local_rotation[name][0] = self.ui.meshZRot.value()
            self.hydro.calculate_transformation()

    def mesh_select(self):
        """Changes values when a different mesh is selected from the drop-down box"""
        name = self.ui.selectMesh.currentText()
        try:
            self.ui.meshName.setText(name)
            self.ui.meshXPos.setValue(self.hydro.local_position[name][0])
            self.ui.meshYPos.setValue(self.hydro.local_position[name][1])
            self.ui.meshZPos.setValue(self.hydro.local_position[name][2])
            self.ui.meshXRot.setValue(self.hydro.local_rotation[name][0])
            self.ui.meshYRot.setValue(self.hydro.local_rotation[name][1])
            self.ui.meshZRot.setValue(self.hydro.local_rotation[name][2])
            self.ui.activeMesh.setChecked(self.hydro.active_mesh[name])
            self.ui.showMesh.setChecked(self.hydro.show_mesh[name])
        except KeyError:
            pass

    def mesh_active(self):
        """Toggles the active state of the mesh"""
        name = self.ui.selectMesh.currentText()
        if name:
            self.hydro.active_mesh[name] = not self.hydro.active_mesh[name]

    def mesh_shown(self):
        """Toggles the shown state of the mesh"""
        name = self.ui.selectMesh.currentText()
        if name:
            self.hydro.show_mesh[name] = not self.hydro.show_mesh[name]

    def add_weight(self):
        """Adds a weight to the buoyancy model"""
        self.hydro.set_weight_force(
            str(len(self.hydro.weight_forces)), np.array([0, 0, 0]), 0
        )
        self.reset_weights()
        self.ui.selectWeight.setCurrentText(str(len(self.hydro.weight_forces)))

    def weight_remove(self):
        old = self.ui.selectWeight.currentText()
        if old:
            self.hydro.weight_forces.pop(old)
            self.hydro.active_weight.pop(old)
            self.hydro.show_weight.pop(old)
            self.reset_weights()
            self.hydro.run()

    def weight_rename(self):
        """Renames the currently selected weight"""
        old = self.ui.selectWeight.currentText()
        if old:
            new = self.ui.weightName.text()
            self.hydro.rename_weight(old, new)
            self.reset_weights()
            self.ui.selectWeight.setCurrentText(new)

    def weight_select(self):
        """Changes values when a different weight is selected from the dropdown box"""
        name = self.ui.selectWeight.currentText()
        try:
            self.ui.weightName.setText(name)
            self.ui.weightXPos.setValue(self.hydro.weight_forces[name][0][0])
            self.ui.weightYPos.setValue(self.hydro.weight_forces[name][0][1])
            self.ui.weightZPos.setValue(self.hydro.weight_forces[name][0][2])
            self.ui.weightMagnitude.setValue(self.hydro.weight_forces[name][1])
            self.ui.activeWeight.setChecked(self.hydro.active_weight[name])
            self.ui.showWeight.setChecked(self.hydro.show_weight[name])
        except KeyError:
            pass

    def weight_x_pos_change(self):
        """Changes the x position of the currently selected weight"""
        name = self.ui.selectWeight.currentText()
        if name:
            self.hydro.weight_forces[name][0][0] = self.ui.weightXPos.value()

    def weight_y_pos_change(self):
        """Changes the x position of the currently selected weight"""
        name = self.ui.selectWeight.currentText()
        if name:
            self.hydro.weight_forces[name][0][1] = self.ui.weightYPos.value()

    def weight_z_pos_change(self):
        """Changes the x position of the currently selected weight"""
        name = self.ui.selectWeight.currentText()
        if name:
            self.hydro.weight_forces[name][0][2] = self.ui.weightZPos.value()

    def weight_magnitude_change(self):
        """Changes the x position of the currently selected weight"""
        name = self.ui.selectWeight.currentText()
        if name:
            self.hydro.weight_forces[name][1] = self.ui.weightMagnitude.value()

    def weight_active(self):
        """Toggles the active state of the currently selected weight"""
        name = self.ui.selectWeight.currentText()
        if name:
            self.hydro.active_weight[name] = not self.hydro.active_weight[name]

    def weight_shown(self):
        """Toggles the shown state of the currently selected weight"""
        name = self.ui.selectWeight.currentText()
        if name:
            self.hydro.show_weight[name] = not self.hydro.show_weight[name]

    def optimize(self):
        """Calls the current solver with the current parameters

        Runs the computation in a seperate thread
        """
        self.hydro.water_density = (
            self.ui.waterRho.value() * 10 ** self.ui.densityPower.value()
        )
        self.hydro.g = self.ui.gravity.value()
        self.hydro.heel = self.ui.globalHeel.value()
        self.hydro.trim = self.ui.globalTrim.value()
        self.hydro.waterplane_origin[0] = self.ui.originX.value()
        self.hydro.waterplane_origin[1] = self.ui.originY.value()
        self.hydro.waterplane_origin[2] = self.ui.originZ.value()
        self.hydro.include_interior = not self.ui.cullInteriorFaces.isChecked()
        self.ui.buttonOptimize.setDisabled(True)

        self.blockSignals(True)
        self.w = Worker(
            solvers[self.ui.selectOptimizer.currentIndex()],
            self.hydro,
            selected=(
                self.ui.heelActive.isChecked(),
                self.ui.trimActive.isChecked(),
                self.ui.heightActive.isChecked(),
            ),
            tol=(
                10 ** self.ui.heelTol.value(),
                10 ** self.ui.trimTol.value(),
                10 ** self.ui.heightTol.value(),
                10 ** self.ui.mxTol.value(),
                10 ** self.ui.myTol.value(),
                10 ** self.ui.fxTol.value(),
            ),
            max_iter=self.ui.maxIter.value(),
            max_time=self.ui.maxTime.value(),
            bounds=[
                [self.ui.heelLB.value(), self.ui.heelUB.value()],
                [self.ui.trimLB.value(), self.ui.trimUB.value()],
                [self.ui.heightLB.value(), self.ui.heightUB.value()],
            ],
        )
        self.w.signals.finished.connect(self.finished_optim)
        self.thread_pool.start(self.w)
        """
        solvers[self.ui.selectOptimizer.currentIndex()](
            self.hydro,
            selected=(self.ui.heelActive.isChecked(), self.ui.trimActive.isChecked(), self.ui.heightActive.isChecked()),
            tol=(10**self.ui.heelTol.value(), 10**self.ui.trimTol.value(), 10**self.ui.heightTol.value(), 10**self.ui.mxTol.value(), 10**self.ui.myTol.value(), 10**self.ui.fxTol.value()),
            max_iter=self.ui.maxIter.value(),
            max_time=self.ui.maxTime.value(),
            bounds=[
                [self.ui.heelLB.value(), self.ui.heelUB.value()],
                [self.ui.trimLB.value(), self.ui.trimUB.value()],
                [self.ui.heightLB.value(), self.ui.heightUB.value()]
            ]
        )
        self.finished_optim()
        """

    def finished_optim(self):
        """Updates display and prints results

        Called when the computation thread finishes running

        Warning
        -------
        This is potentially bugged. If more than two threads are running, then blockSignals may cause unusual results.
        Also, the model and results displayed are often incorrect until the result is recalculated.
        """
        self.blockSignals(False)
        self.ui.buttonOptimize.setDisabled(False)
        self.update_window()
        self.ui.resultsOutput.setHtml(self.html())
        self.reset_globals()

    def recalculate(self):
        """Performs a full calculation of buoyancy forces and hydrostatic particulars"""
        self.hydro.water_density = (
            self.ui.waterRho.value() * 10 ** self.ui.densityPower.value()
        )
        self.hydro.g = self.ui.gravity.value()
        self.hydro.heel = self.ui.globalHeel.value()
        self.hydro.trim = self.ui.globalTrim.value()
        self.hydro.waterplane_origin[0] = self.ui.originX.value()
        self.hydro.waterplane_origin[1] = self.ui.originY.value()
        self.hydro.waterplane_origin[2] = self.ui.originZ.value()
        self.hydro.include_interior = not self.ui.cullInteriorFaces.isChecked()

        self.hydro.calculate_results()
        self.ui.resultsOutput.setHtml(self.html())

    def update_window(self):
        """Updates the 3D viewport"""
        reference = "body"
        if self.ui.earthCoordinates.isChecked():
            reference = "earth"
        self.vtk_widget.clear()
        if self.ui.showFullMesh.isChecked():
            for mesh in self.hydro.plot_transformed(reference):
                self.vtk_widget.add_mesh(mesh, color=[1, 0, 0], opacity=0.9)
        if self.ui.showCutMesh.isChecked():
            for mesh in self.hydro.plot_below_surface(reference):
                self.vtk_widget.add_mesh(mesh, color=[0, 0, 1], opacity=0.9)
        if self.ui.showAllWeights.isChecked():
            for mesh in self.hydro.plot_weights(reference):
                self.vtk_widget.add_mesh(mesh, color=[0.1, 0.1, 0.1], opacity=1)
        if self.ui.showWaterplane.isChecked():
            self.vtk_widget.add_mesh(
                self.hydro.plot_water_plane(reference),
                color=[0, 0.2, 1],
                opacity=0.9,
            )
            self.vtk_widget.remove_scalar_bar()
        self.vtk_widget.add_mesh(self.hydro.plot_bounding_box(reference), opacity=0.3)
        if self.ui.showCentres.isChecked():
            self.vtk_widget.add_point_labels(
                self.hydro.plot_centres(reference),
                "Labels",
                point_size=20,
                font_size=24,
                render_points_as_spheres=True,
            )
        self.vtk_widget.show_grid()

    def reset_meshes(self):
        """Clears the dropdown box and re-adds the currently loaded meshes"""
        self.ui.selectMesh.clear()
        self.ui.selectMesh.addItems(self.hydro.meshes.keys())

        disable = len(self.hydro.meshes) == 0
        self.ui.meshName.setDisabled(disable)
        self.ui.meshXPos.setDisabled(disable)
        self.ui.meshYPos.setDisabled(disable)
        self.ui.meshZPos.setDisabled(disable)
        self.ui.meshXRot.setDisabled(disable)
        self.ui.meshYRot.setDisabled(disable)
        self.ui.meshZRot.setDisabled(disable)

    def reset_weights(self):
        """Clears the dropdown box and re-adds the currently loaded weights"""
        self.ui.selectWeight.clear()
        self.ui.selectWeight.addItems(self.hydro.weight_forces.keys())

        disable = len(self.hydro.weight_forces) == 0
        self.ui.weightName.setDisabled(disable)
        self.ui.weightXPos.setDisabled(disable)
        self.ui.weightYPos.setDisabled(disable)
        self.ui.weightZPos.setDisabled(disable)
        self.ui.weightMagnitude.setDisabled(disable)

    def reset_globals(self):
        """Resets the waterplane origin, heel and trim values to those in the buoyancy model"""
        self.ui.originX.setValue(self.hydro.waterplane_origin[0])
        self.ui.originY.setValue(self.hydro.waterplane_origin[1])
        self.ui.originZ.setValue(self.hydro.waterplane_origin[2])

        self.ui.globalHeel.setValue(self.hydro.heel)
        self.ui.globalTrim.setValue(self.hydro.trim)

    @Qt.pyqtSlot(str)
    def append_text(self, text):
        """Appends text to the gui stdout"""
        self.ui.winStdOut.moveCursor(Qt.QTextCursor.End)
        self.ui.winStdOut.insertPlainText(text)

    def print(self):
        """Prints the current document in the results viewer to a PDF"""
        name = Qt.QFileDialog.getSaveFileName(self, "Save Report", filter="PDF (*.pdf)")
        if name:
            self.ui.resultsOutput.page().printToPdf(
                name[0],
                pageLayout=Qt.QPageLayout(
                    Qt.QPageSize(Qt.QPageSize.A4),
                    Qt.QPageLayout.Portrait,
                    Qt.QMarginsF(20, 20, 20, 20),
                ),
            )

    def report(self):
        """A full html report including titles and images

        Returns
        -------
        str
        """
        self.update_window()
        im = Image.fromarray(self.vtk_widget.image.astype("uint8"))
        rawBytes = io.BytesIO()
        im.save(rawBytes, "PNG")
        rawBytes.seek(0)
        im = base64.b64encode(rawBytes.read())

        head = (
            """
<head>
<link rel="stylesheet" href="""
            + os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "gui_src", "github-markdown.css"
                )
            )
            + """>
<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full">
</script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            processEscapes: true
        },
        config: ["MMLorHTML.js"],
        jax: ["input/TeX", "output/HTML-CSS", "output/NativeMML"],
        extensions: ["MathMenu.js", "MathZoom.js"]
    });
</script>
</head>
<style>
    .markdown-body {
        box-sizing: border-box;
        min-width: 200px;
        max-width: 980px;
        margin: 0 auto;
        padding: 45px;
    }

    @media (max-width: 767px) {
        .markdown-body {
            padding: 15px;
        }
    }
</style>
<article class="markdown-body">
"""
        )
        title = markdown.Markdown(
            extensions=[
                "mdx_math",
                "markdown.extensions.tables",
            ]
        ).convert(
            f"""
# Hydrostatics Report {datetime.datetime.now()}

**Weights:** {', '.join(name for name,active in self.hydro.active_weight.items() if active)}

**Meshes:** {', '.join(name for name,active in self.hydro.active_mesh.items() if active)}

**Waterplane Origin:** {self.hydro.waterplane_origin}

**Heel:** {self.hydro.heel:.2f}, **Error:** {self.hydro.results.GZ_t:.2e} mm

**Trim:** {self.hydro.trim:.2f}, **Error:** {self.hydro.results.GZ_l:.2e} mm

**Volume Error:** {self.hydro.results.volume_error:.2e} mm^3, **Percentage:** {self.hydro.results.volume_error/self.hydro.results.volume*100 if self.hydro.results.volume > 0 else 0.0:.2e} %

"""
        )

        image = f"""
<img src="data:image/png;base64, {im.decode('utf-8')}" alt="Screenshot" />
<div style="page-break-after: always;"></div>
"""
        res = (
            head
            + title
            + image
            + markdown.Markdown(
                extensions=[
                    "mdx_math",
                    "markdown.extensions.tables",
                ]
            ).convert(self.hydro.results.markdown())
            + "</article>"
        )
        self.ui.resultsOutput.setHtml(res)

    def html(self):
        """A minimal html representation of the current results

        Returns
        -------
        str
        """
        head = (
            """
<head>
<link rel="stylesheet" href="""
            + os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "gui_src", "github-markdown.css"
                )
            )
            + """>
<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full">
</script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            processEscapes: true
        },
        config: ["MMLorHTML.js"],
        jax: ["input/TeX", "output/HTML-CSS", "output/NativeMML"],
        extensions: ["MathMenu.js", "MathZoom.js"]
    });
</script>
</head>
<style>
    .markdown-body {
        box-sizing: border-box;
        min-width: 200px;
        max-width: 980px;
        margin: 0 auto;
        padding: 45px;
    }

    @media (max-width: 767px) {
        .markdown-body {
            padding: 15px;
        }
    }
</style>
<article class="markdown-body">
"""
        )
        res = (
            head
            + markdown.Markdown(
                extensions=[
                    "mdx_math",
                    "markdown.extensions.tables",
                ]
            ).convert(self.hydro.results.markdown())
            + "</article>"
        )
        return res


class LoadPopup(Qt.QDialog):
    """A popup window that asks for the shape of the UV mesh when loaded"""

    def __init__(self, parent=None):
        super().__init__(parent)

        shape = Qt.QLabel(self)
        shape.setText("Shape:")

        self.dim1 = Qt.QSpinBox(self)
        self.dim2 = Qt.QSpinBox(self)
        self.checkMirror = Qt.QCheckBox(self)
        self.checkMirror.setText("Mirror Mesh")
        self.checkMirror.setChecked(True)
        self.checkClose = Qt.QCheckBox(self)
        self.checkClose.setText("Close Mesh")
        self.checkClose.setChecked(True)
        button = Qt.QDialogButtonBox(
            Qt.QDialogButtonBox.Ok | Qt.QDialogButtonBox.Cancel, self
        )

        layout = Qt.QGridLayout(self)
        layout.addWidget(shape, 0, 0, 1, 2)
        layout.addWidget(self.dim1, 1, 0)
        layout.addWidget(self.dim2, 1, 1)
        layout.addWidget(self.checkMirror, 2, 0)
        layout.addWidget(self.checkClose, 2, 1)
        layout.addWidget(button, 3, 0, 1, 2)

        button.accepted.connect(self.accept)
        button.rejected.connect(self.reject)

        self.setWindowTitle("DXF Load Parameters")
        self.setWindowIcon(
            Qt.QIcon(
                os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "gui_src", "logo.png")
                )
            )
        )

    def getInputs(self):
        return (
            self.dim1.value(),
            self.dim2.value(),
            self.checkMirror.isChecked(),
            self.checkClose.isChecked(),
        )


class WriteStream:
    """Basic writestream

    Can be used to replace stdout
    """

    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass


class Receiver(Qt.QObject):
    """Gets text from queue and emits it

    Used for threaded redirection of stdout
    """

    s = Qt.pyqtSignal(str)

    def __init__(self, queue, *args, **kwargs):
        Qt.QObject.__init__(self, *args, **kwargs)
        self.queue = queue

    @Qt.pyqtSlot()
    def run(self):
        while True:
            text = self.queue.get()
            if text == "EXIT_WINSTDOUT":
                break
            else:
                self.s.emit(text)


def hydro():
    """Main runner for gui

    Exported to command line
    """
    # This is required to get the window icon working properly
    import ctypes

    logging.info("Started")

    myappid = "sailgp.hydrostatics.gui.001"  # arbitrary string
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # Start main app
    sys.argv.append("--disable-web-security")
    app = Qt.QApplication(sys.argv)
    app.setStyle("Fusion")
    logging.info("Loading Window")
    window = MainWindow()
    logging.info("Loaded Window")

    # Redirect stdout to window where possible
    thread = Qt.QThread()
    queue = Queue()
    backup = sys.stdout
    sys.stdout = WriteStream(queue)
    r = Receiver(queue)
    r.s.connect(window.append_text)
    r.moveToThread(thread)
    thread.started.connect(r.run)
    thread.start()
    logging.info("Started threads")

    res = app.exec_()
    print("EXIT_WINSTDOUT")
    sys.stdout = backup
    thread.exit()
    thread.wait()

    del window
    del app

    return res


if __name__ == "__main__":
    hydro()
