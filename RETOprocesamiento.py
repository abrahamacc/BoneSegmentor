import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Slider, Button
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure, morphology
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os

class BoneSegmenter:
    def __init__(self):
        self.dicom_data = None
        self.segmented_mask = None
        # Stores (plane_type, slice_idx, y1, y2, x1, x2) for the selected 2D region
        self.selected_region = None
        self.hu_threshold_min = 200  # Valor típico para huesos
        self.hu_threshold_max = 3000
        self.current_slice_axial = 0 # Para el plano axial (Z)
        self.current_slice_coronal = 0 # Para el plano coronal (Y)
        self.current_slice_sagital = 0 # Para el plano sagital (X)
        self.region_selected = False
        
        # Atributos para los sliders y figuras de Matplotlib
        self.fig_selection = None
        self.ax_axial = None
        self.ax_coronal = None
        self.ax_sagital = None
        self.img_axial = None
        self.img_coronal = None
        self.img_sagital = None
        self.slider_axial = None
        self.slider_coronal = None
        self.slider_sagital = None
        self.selector = None # Mantener el selector de rectángulo aquí
        self.active_selector_plane = 'axial' # 'axial', 'coronal', 'sagital'

    def load_dicom_series(self, folder_path):
        """Carga una serie DICOM desde una carpeta"""
        try:
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(folder_path)
            
            if not series_ids:
                raise ValueError("No se encontraron series DICOM en la carpeta")
            
            # Usar la primera serie encontrada
            series_id = series_ids[0]
            dicom_names = reader.GetGDCMSeriesFileNames(folder_path, series_id)
            reader.SetFileNames(dicom_names)
            
            self.dicom_data = reader.Execute()
            
            size = self.dicom_data.GetSize() # (X, Y, Z)
            self.current_slice_axial = size[2] // 2
            self.current_slice_coronal = size[1] // 2
            self.current_slice_sagital = size[0] // 2

            print(f"DICOM cargado: {self.dicom_data.GetSize()} (X, Y, Z)")
            return True
            
        except Exception as e:
            print(f"Error cargando DICOM: {e}")
            messagebox.showerror("Error de Carga", f"No se pudo cargar el DICOM: {e}")
            return False
    
    def load_single_dicom(self, file_path):
        """Carga un archivo DICOM individual"""
        try:
            self.dicom_data = sitk.ReadImage(file_path)
            
            size = self.dicom_data.GetSize() # (X, Y, Z)
            if len(size) == 2: # Si es 2D, asumimos Z=1
                # Convert 2D image to a 3D image with a single slice
                self.dicom_data = sitk.GetImageFromArray(sitk.GetArrayFromImage(self.dicom_data)[np.newaxis, :, :])
                size = self.dicom_data.GetSize() # Update size after conversion

            self.current_slice_axial = size[2] // 2
            self.current_slice_coronal = size[1] // 2
            self.current_slice_sagital = size[0] // 2

            print(f"DICOM cargado: {self.dicom_data.GetSize()} (X, Y, Z)")
            return True
        except Exception as e:
            print(f"Error cargando DICOM: {e}")
            messagebox.showerror("Error de Carga", f"No se pudo cargar el DICOM: {e}")
            return False
    
    def select_region_first(self):
        """Permite al usuario seleccionar una región específica ANTES de segmentar, con navegación 3D y selección de plano."""
        if self.dicom_data is None:
            messagebox.showwarning("Advertencia", "Primero debes cargar un archivo DICOM.")
            return None
        
        # Cerrar cualquier figura previa de selección para evitar múltiples instancias
        if self.fig_selection is not None and plt.fignum_exists(self.fig_selection.number):
            plt.close(self.fig_selection)

        original_array_sitk = self.dicom_data
        image_array_zyx = sitk.GetArrayFromImage(original_array_sitk) # Z, Y, X
        
        num_slices_z, height_y, width_x = image_array_zyx.shape
        
        # Ensure current slice indices are within bounds
        if self.current_slice_axial >= num_slices_z: self.current_slice_axial = num_slices_z // 2
        if self.current_slice_coronal >= height_y: self.current_slice_coronal = height_y // 2
        if self.current_slice_sagital >= width_x: self.current_slice_sagital = width_x // 2
        
        self.fig_selection, axs = plt.subplots(1, 3, figsize=(18, 6))
        self.ax_axial, self.ax_coronal, self.ax_sagital = axs
        
        # --- Vista Axial (Z) ---
        self.img_axial = self.ax_axial.imshow(image_array_zyx[self.current_slice_axial, :, :], cmap='gray')
        self.ax_axial.set_title(f'Axial (Slice {self.current_slice_axial}/{num_slices_z-1})')
        self.ax_axial.axis('off')
        
        # --- Vista Coronal (Y) ---
        self.img_coronal = self.ax_coronal.imshow(image_array_zyx[:, self.current_slice_coronal, :], cmap='gray')
        self.ax_coronal.set_title(f'Coronal (Slice {self.current_slice_coronal}/{height_y-1})')
        self.ax_coronal.axis('off')
        
        # --- Vista Sagital (X) ---
        self.img_sagital = self.ax_sagital.imshow(image_array_zyx[:, :, self.current_slice_sagital], cmap='gray')
        self.ax_sagital.set_title(f'Sagital (Slice {self.current_slice_sagital}/{width_x-1})')
        self.ax_sagital.axis('off')
        
        plt.tight_layout()

        # --- Sliders para navegar las slices ---
        axcolor = 'lightgoldenrodyellow'
        
        ax_slider_axial = plt.axes([0.1, 0.05, 0.2, 0.03], facecolor=axcolor)
        self.slider_axial = Slider(ax_slider_axial, 'Axial Slice', 0, num_slices_z - 1, valinit=self.current_slice_axial, valstep=1)
        self.slider_axial.on_changed(self._update_axial_slice)

        ax_slider_coronal = plt.axes([0.4, 0.05, 0.2, 0.03], facecolor=axcolor)
        self.slider_coronal = Slider(ax_slider_coronal, 'Coronal Slice', 0, height_y - 1, valinit=self.current_slice_coronal, valstep=1)
        self.slider_coronal.on_changed(self._update_coronal_slice)
        
        ax_slider_sagital = plt.axes([0.7, 0.05, 0.2, 0.03], facecolor=axcolor)
        self.slider_sagital = Slider(ax_slider_sagital, 'Sagital Slice', 0, width_x - 1, valinit=self.current_slice_sagital, valstep=1)
        self.slider_sagital.on_changed(self._update_sagital_slice)

        # --- Botones para seleccionar el plano de selección ---
        ax_btn_axial = plt.axes([0.1, 0.005, 0.08, 0.035])
        btn_axial = Button(ax_btn_axial, 'Select Axial')
        btn_axial.on_clicked(lambda event: self._activate_selector_on_plane('axial'))

        ax_btn_coronal = plt.axes([0.4, 0.005, 0.08, 0.035])
        btn_coronal = Button(ax_btn_coronal, 'Select Coronal')
        btn_coronal.on_clicked(lambda event: self._activate_selector_on_plane('coronal'))
        
        ax_btn_sagital = plt.axes([0.7, 0.005, 0.08, 0.035])
        btn_sagital = Button(ax_btn_sagital, 'Select Sagital')
        btn_sagital.on_clicked(lambda event: self._activate_selector_on_plane('sagital'))

        # Inicializar el selector en el plano axial por defecto
        self._activate_selector_on_plane('axial') # This will also draw existing selection if any
        
        self.fig_selection.canvas.mpl_connect('close_event', self.on_close_selection_fig)
        plt.show(block=False)
        
        return self.fig_selection

    def _activate_selector_on_plane(self, plane):
        """Activa el RectangleSelector en el plano especificado y lo desactiva en los demás."""
        # Deactivate any currently active selector
        if self.selector:
            self.selector.set_active(False)
            self.selector = None # Clear the reference to remove the old selector

        self.active_selector_plane = plane
        
        # Clear previous rectangles from all axes before setting up new selector
        for ax in [self.ax_axial, self.ax_coronal, self.ax_sagital]:
            for p in list(ax.patches):
                p.remove()
            for l in list(ax.lines): # Also clear lines if any, though patches are main for rect
                l.remove()

        # Update titles to indicate which plane is active for selection
        self.ax_axial.set_title(f'Axial (Slice {self.current_slice_axial})')
        self.ax_coronal.set_title(f'Coronal (Slice {self.current_slice_coronal})')
        self.ax_sagital.set_title(f'Sagital (Slice {self.current_slice_sagital})')

        # Activate the selector on the chosen plane
        if plane == 'axial':
            self.selector = RectangleSelector(self.ax_axial, self.on_select_region, useblit=True,
                                            button=[1], minspanx=5, minspany=5, spancoords='pixels',
                                            interactive=True)
            self.ax_axial.set_title(f'Axial (SELECCIONANDO AQUÍ - Slice {self.current_slice_axial})')
        elif plane == 'coronal':
            self.selector = RectangleSelector(self.ax_coronal, self.on_select_region, useblit=True,
                                            button=[1], minspanx=5, minspany=5, spancoords='pixels',
                                            interactive=True)
            self.ax_coronal.set_title(f'Coronal (SELECCIONANDO AQUÍ - Slice {self.current_slice_coronal})')
        elif plane == 'sagital':
            self.selector = RectangleSelector(self.ax_sagital, self.on_select_region, useblit=True,
                                            button=[1], minspanx=5, minspany=5, spancoords='pixels',
                                            interactive=True)
            self.ax_sagital.set_title(f'Sagital (SELECCIONANDO AQUÍ - Slice {self.current_slice_sagital})')
        
        # Redraw selection on all planes after activating the new selector
        self._draw_region_on_plane(self.ax_axial, 'axial', self.current_slice_axial)
        self._draw_region_on_plane(self.ax_coronal, 'coronal', self.current_slice_coronal)
        self._draw_region_on_plane(self.ax_sagital, 'sagital', self.current_slice_sagital)

        self.fig_selection.canvas.draw_idle()
        print(f"Selector activado en plano: {self.active_selector_plane}")

    def _draw_region_on_plane(self, ax, plane_type, slice_idx):
        """
        Dibuja la región seleccionada en el plano especificado.
        Convierte la ROI 2D seleccionada en el plano original a una representación 
        apropiada para el plano actual (axial, coronal, sagital).
        """
        # Clear existing patches on this axis before drawing
        for p in list(ax.patches):
            p.remove()

        if self.selected_region:
            sel_plane, sel_slice_idx, r1, r2, c1, c2 = self.selected_region
            
            # Determine the 3D bounding box (min_x, max_x, min_y, max_y, min_z, max_z)
            # based on the 2D selection and the plane it was made on.
            min_x, max_x, min_y, max_y, min_z, max_z = 0, 0, 0, 0, 0, 0
            
            # Map 2D selection (r1,r2,c1,c2) and sel_slice_idx to 3D volume coordinates (X,Y,Z)
            if sel_plane == 'axial':
                # Axial: r is Y, c is X, slice is Z
                min_y, max_y = r1, r2
                min_x, max_x = c1, c2
                min_z, max_z = sel_slice_idx, sel_slice_idx
            elif sel_plane == 'coronal':
                # Coronal: r is Z, c is X, slice is Y
                min_z, max_z = r1, r2
                min_x, max_x = c1, c2
                min_y, max_y = sel_slice_idx, sel_slice_idx
            elif sel_plane == 'sagital':
                # Sagital: r is Z, c is Y, slice is X
                min_z, max_z = r1, r2
                min_y, max_y = c1, c2
                min_x, max_x = sel_slice_idx, sel_slice_idx
            
            # Project the 3D bounding box onto the current display plane
            if plane_type == 'axial' and slice_idx == self.current_slice_axial:
                if sel_plane == 'axial':
                    rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                         fill=False, edgecolor='yellow', linewidth=2)
                else:
                    if min_z <= slice_idx <= max_z:
                        rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                             fill=False, edgecolor='cyan', linewidth=1, linestyle='--')
                    else: return 
                ax.add_patch(rect)

            elif plane_type == 'coronal' and slice_idx == self.current_slice_coronal:
                if sel_plane == 'coronal':
                    rect = plt.Rectangle((min_x, min_z), max_x - min_x, max_z - min_z,
                                         fill=False, edgecolor='yellow', linewidth=2)
                else: 
                    if min_y <= slice_idx <= max_y:
                        rect = plt.Rectangle((min_x, min_z), max_x - min_x, max_z - min_z,
                                             fill=False, edgecolor='cyan', linewidth=1, linestyle='--')
                    else: return 
                ax.add_patch(rect)

            elif plane_type == 'sagital' and slice_idx == self.current_slice_sagital:
                if sel_plane == 'sagital':
                    rect = plt.Rectangle((min_y, min_z), max_y - min_y, max_z - min_z,
                                         fill=False, edgecolor='yellow', linewidth=2)
                else: 
                    if min_x <= slice_idx <= max_x:
                        rect = plt.Rectangle((min_y, min_z), max_y - min_y, max_z - min_z,
                                             fill=False, edgecolor='cyan', linewidth=1, linestyle='--')
                    else: return 
                ax.add_patch(rect)
                
    def _update_axial_slice(self, val):
        slice_idx = int(val)
        self.current_slice_axial = slice_idx
        image_array_zyx = sitk.GetArrayFromImage(self.dicom_data)
        self.img_axial.set_data(image_array_zyx[slice_idx, :, :])
        
        if self.active_selector_plane == 'axial':
            self.ax_axial.set_title(f'Axial (SELECCIONANDO AQUÍ - Slice {slice_idx}/{image_array_zyx.shape[0]-1})')
        else:
            self.ax_axial.set_title(f'Axial (Slice {slice_idx}/{image_array_zyx.shape[0]-1})')
        
        for p in list(self.ax_axial.patches):
            p.remove()
        self._draw_region_on_plane(self.ax_axial, 'axial', slice_idx)
        
        self.fig_selection.canvas.draw_idle()

    def _update_coronal_slice(self, val):
        slice_idx = int(val)
        self.current_slice_coronal = slice_idx
        image_array_zyx = sitk.GetArrayFromImage(self.dicom_data)
        self.img_coronal.set_data(image_array_zyx[:, slice_idx, :])

        if self.active_selector_plane == 'coronal':
            self.ax_coronal.set_title(f'Coronal (SELECCIONANDO AQUÍ - Slice {slice_idx}/{image_array_zyx.shape[1]-1})')
        else:
            self.ax_coronal.set_title(f'Coronal (Slice {slice_idx}/{image_array_zyx.shape[1]-1})')

        for p in list(self.ax_coronal.patches):
            p.remove()
        self._draw_region_on_plane(self.ax_coronal, 'coronal', slice_idx)

        self.fig_selection.canvas.draw_idle()

    def _update_sagital_slice(self, val):
        slice_idx = int(val)
        self.current_slice_sagital = slice_idx
        image_array_zyx = sitk.GetArrayFromImage(self.dicom_data)
        self.img_sagital.set_data(image_array_zyx[:, :, slice_idx])

        if self.active_selector_plane == 'sagital':
            self.ax_sagital.set_title(f'Sagital (SELECCIONANDO AQUÍ - Slice {slice_idx}/{image_array_zyx.shape[2]-1})')
        else:
            self.ax_sagital.set_title(f'Sagital (Slice {slice_idx}/{image_array_zyx.shape[2]-1})')
            
        for p in list(self.ax_sagital.patches):
            p.remove()
        self._draw_region_on_plane(self.ax_sagital, 'sagital', slice_idx)
            
        self.fig_selection.canvas.draw_idle()

    def on_close_selection_fig(self, event):
        """Callback cuando se cierra la figura de selección."""
        print("Ventana de selección de región cerrada.")
        # Make sure to deactivate the selector when the window closes to prevent lingering events
        if self.selector:
            self.selector.set_active(False)
            self.selector = None
    
    def on_select_region(self, eclick, erelease):
        """Callback para selección de región inicial. Ahora captura el plano y la slice actual."""
        x1_data, y1_data = int(eclick.xdata), int(eclick.ydata)
        x2_data, y2_data = int(erelease.xdata), int(erelease.ydata)
        
        # Asegurar orden correcto
        x1_data, x2_data = min(x1_data, x2_data), max(x1_data, x2_data)
        y1_data, y2_data = min(y1_data, y2_data), max(y1_data, y2_data)
        
        # Guardar la región seleccionada con el plano y la slice correspondiente
        if self.active_selector_plane == 'axial':
            # In axial (Z): y_data is Y, x_data is X. Z is current_slice_axial
            self.selected_region = ('axial', self.current_slice_axial, y1_data, y2_data, x1_data, x2_data)
            print(f"Región seleccionada en Axial: Slice Z: {self.current_slice_axial}, Y: {y1_data}-{y2_data}, X: {x1_data}-{x2_data}")
        elif self.active_selector_plane == 'coronal':
            # In coronal (Y): y_data is Z, x_data is X. Y is current_slice_coronal
            self.selected_region = ('coronal', self.current_slice_coronal, y1_data, y2_data, x1_data, x2_data)
            print(f"Región seleccionada en Coronal: Slice Y: {self.current_slice_coronal}, Z: {y1_data}-{y2_data}, X: {x1_data}-{x2_data}")
        elif self.active_selector_plane == 'sagital':
            # In sagital (X): y_data is Z, x_data is Y. X is current_slice_sagital
            self.selected_region = ('sagital', self.current_slice_sagital, y1_data, y2_data, x1_data, x2_data)
            print(f"Región seleccionada en Sagital: Slice X: {self.current_slice_sagital}, Z: {y1_data}-{y2_data}, Y: {x1_data}-{x2_data}")
            
        self.region_selected = True
        
        # Redraw the selection on all planes to show the newly selected region
        self._draw_region_on_plane(self.ax_axial, 'axial', self.current_slice_axial)
        self._draw_region_on_plane(self.ax_coronal, 'coronal', self.current_slice_coronal)
        self._draw_region_on_plane(self.ax_sagital, 'sagital', self.current_slice_sagital)
        self.fig_selection.canvas.draw_idle()

        # Get and print HU value at the calculated seed point for debugging
        if self.dicom_data:
            image_array_zyx = sitk.GetArrayFromImage(self.dicom_data)
            seed_x, seed_y, seed_z = 0, 0, 0
            if self.active_selector_plane == 'axial':
                seed_x = (x1_data + x2_data) // 2
                seed_y = (y1_data + y2_data) // 2
                seed_z = self.current_slice_axial
            elif self.active_selector_plane == 'coronal':
                seed_x = (x1_data + x2_data) // 2
                seed_y = self.current_slice_coronal
                seed_z = (y1_data + y2_data) // 2
            elif self.active_selector_plane == 'sagital':
                seed_x = self.current_slice_sagital
                seed_y = (y1_data + y2_data) // 2
                seed_z = (x1_data + x2_data) // 2 # x_data in sagital is Y, y_data is Z

            # Adjust for ZYX to XYZ indexing for printing HU value if needed
            if 0 <= seed_z < image_array_zyx.shape[0] and \
               0 <= seed_y < image_array_zyx.shape[1] and \
               0 <= seed_x < image_array_zyx.shape[2]:
                hu_at_seed = image_array_zyx[seed_z, seed_y, seed_x]
                print(f"HU value at seed point ({seed_x}, {seed_y}, {seed_z}) in {self.active_selector_plane} plane: {hu_at_seed}")
            else:
                print(f"Warning: Seed point ({seed_x}, {seed_y}, {seed_z}) is out of bounds for HU value check.")
    
    def segment_bones_in_region(self, min_hu=None, max_hu=None):
        """Segmenta huesos SOLO en la región seleccionada, usando la información 3D del plano y slice."""
        if self.dicom_data is None:
            messagebox.showwarning("Advertencia", "Primero debes cargar un archivo DICOM.")
            return False
        
        if not self.region_selected or self.selected_region is None:
            messagebox.showwarning("Advertencia", "Primero debes seleccionar una región.")
            return False
        
        min_hu = min_hu or self.hu_threshold_min
        max_hu = max_hu or self.hu_threshold_max
        
        # Desempaquetar la región seleccionada
        plane_type, slice_idx_selected, r1, r2, c1, c2 = self.selected_region
        
        # Convertir a SimpleITK Image
        image_sitk = self.dicom_data
        
        # Determinar la semilla (X, Y, Z) para SimpleITK
        seed_x, seed_y, seed_z = 0, 0, 0
        
        # SimpleITK image.GetSize() returns (X, Y, Z) for dimensions
        # SimpleITK.GetArrayFromImage() returns numpy array with shape (Z, Y, X)
        # So we need to map our 2D selection coordinates (r, c) to the correct (X, Y, Z) for the seed.
        image_array = sitk.GetArrayFromImage(image_sitk)  # (Z, Y, X)

        # Ajusta los límites para asegurar que están dentro del volumen
        z_max, y_max, x_max = image_array.shape
        r1, r2 = max(0, r1), min(y_max, r2)
        c1, c2 = max(0, c1), min(x_max, c2)

        # Extraer subvolumen de la región seleccionada
        if plane_type == 'axial':
            subregion = image_array[slice_idx_selected, r1:r2, c1:c2]
        elif plane_type == 'coronal':
            subregion = image_array[r1:r2, slice_idx_selected, c1:c2]
        elif plane_type == 'sagital':
            subregion = image_array[r1:r2, c1:c2, slice_idx_selected]

        # Buscar la coordenada con mayor HU dentro del subvolumen (idealmente hueso)
        max_idx = np.unravel_index(np.argmax(subregion), subregion.shape)
        max_value = subregion[max_idx]

        if plane_type == 'axial':
            seed_y, seed_x = max_idx
            seed_y += r1
            seed_x += c1
            seed_z = slice_idx_selected
        elif plane_type == 'coronal':
            seed_z, seed_x = max_idx
            seed_z += r1
            seed_x += c1
            seed_y = slice_idx_selected
        elif plane_type == 'sagital':
            seed_z, seed_y = max_idx
            seed_z += r1
            seed_y += c1
            seed_x = slice_idx_selected

        print(f"Semilla corregida para ConnectedThreshold ({plane_type}): X={seed_x}, Y={seed_y}, Z={seed_z}, HU={max_value}")

        # Check if seed is within image bounds
        img_size = image_sitk.GetSize() # (X, Y, Z)
        if not (0 <= seed_x < img_size[0] and 0 <= seed_y < img_size[1] and 0 <= seed_z < img_size[2]):
            messagebox.showerror("Error de Semilla", f"La semilla de la región seleccionada ({seed_x}, {seed_y}, {seed_z}) está fuera de los límites de la imagen {img_size}. Por favor, selecciona una región válida.")
            return False

        
        # Asegurarse de que la imagen sea float para ConnectedThreshold
        image_sitk_float = sitk.Cast(image_sitk, sitk.sitkFloat32)

        # Crear el filtro
        connected_threshold_filter = sitk.ConnectedThresholdImageFilter()
        connected_threshold_filter.SetLower(float(min_hu))
        connected_threshold_filter.SetUpper(float(max_hu))

        # Añadir la semilla (convertida a enteros normales)
        connected_threshold_filter.AddSeed([int(seed_x), int(seed_y), int(seed_z)])
        
        try:
            segmented_image_sitk = connected_threshold_filter.Execute(image_sitk_float)
            self.segmented_mask = sitk.GetArrayFromImage(segmented_image_sitk).astype(np.uint8)
            
            # Post-processing only if some voxels are found
            if np.sum(self.segmented_mask) > 0:
                self.segmented_mask = morphology.remove_small_objects(self.segmented_mask.astype(bool), min_size=50, connectivity=3)
                self.segmented_mask = self.segmented_mask.astype(np.uint8)
                self.segmented_mask = morphology.binary_closing(self.segmented_mask, morphology.ball(1))
            
            print(f"Segmentación de hueso específico completada. Voxels de hueso: {np.sum(self.segmented_mask)}")
            if np.sum(self.segmented_mask) == 0:
                messagebox.showwarning("Segmentación Regional", "No se encontró ningún hueso en la región seleccionada con los parámetros actuales. Intenta ajustar el rango HU o la selección de la región/semilla.")
            else:
                messagebox.showinfo("Segmentación", "Segmentación de región completada exitosamente.")
            return True
        except Exception as e:
            print(f"Error en ConnectedThreshold: {e}")
            messagebox.showerror("Error de Segmentación", f"No se pudo segmentar la región: {e}")
            return False

    def segment_bones_full(self, min_hu=None, max_hu=None):
        """Segmenta huesos en toda la imagen (método original)"""
        if self.dicom_data is None:
            messagebox.showwarning("Advertencia", "Primero debes cargar un archivo DICOM.")
            return False
        
        min_hu = min_hu or self.hu_threshold_min
        max_hu = max_hu or self.hu_threshold_max
        
        # Convertir a array numpy (Z, Y, X)
        image_array_zyx = sitk.GetArrayFromImage(self.dicom_data)
        
        # Aplicar threshold para huesos
        bone_mask = (image_array_zyx >= min_hu) & (image_array_zyx <= max_hu)
        
        # Limpiar ruido con operaciones morfológicas 3D
        bone_mask = morphology.remove_small_objects(bone_mask, min_size=100, connectivity=3)
        bone_mask = morphology.binary_closing(bone_mask, morphology.ball(1))
        
        self.segmented_mask = bone_mask.astype(np.uint8)
        print(f"Segmentación completa completada. Voxels de hueso: {np.sum(self.segmented_mask)}")
        messagebox.showinfo("Segmentación", "Segmentación de imagen completa completada exitosamente.")
        return True
    
    def show_segmentation_result(self):
        """Muestra el resultado de la segmentación"""
        if self.segmented_mask is None:
            messagebox.showwarning("Advertencia", "No hay segmentación para mostrar.")
            return
        
        original_array_zyx = sitk.GetArrayFromImage(self.dicom_data)
        
        # Usar la slice axial seleccionada previamente (o el centro) para la visualización del resultado
        slice_to_display_axial = self.current_slice_axial 
        if slice_to_display_axial >= original_array_zyx.shape[0]:
            slice_to_display_axial = original_array_zyx.shape[0] // 2 # Fallback
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Imagen original
        ax1.imshow(original_array_zyx[slice_to_display_axial], cmap='gray')
        ax1.set_title(f'Imagen Original (Slice Axial {slice_to_display_axial})')
        ax1.axis('off')
        
        # Máscara de huesos superpuesta
        ax2.imshow(original_array_zyx[slice_to_display_axial], cmap='gray', alpha=0.3) 
        ax2.imshow(self.segmented_mask[slice_to_display_axial], cmap='Reds', alpha=0.7)
        ax2.set_title(f'Huesos Segmentados (Slice Axial {slice_to_display_axial})')
        ax2.axis('off')
        
        # Marcar región seleccionada si existe (en la slice axial original de selección)
        if self.selected_region and self.selected_region[0] == 'axial' and self.selected_region[1] == slice_to_display_axial:
            _, _, y1, y2, x1, x2 = self.selected_region
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                 fill=False, edgecolor='yellow', linewidth=2)
            ax2.add_patch(rect)
        elif self.selected_region: # Show projection of 3D ROI if current slice is not the selection plane
            sel_plane, sel_slice_idx, r1, r2, c1, c2 = self.selected_region
            min_x, max_x, min_y, max_y, min_z, max_z = 0,0,0,0,0,0

            if sel_plane == 'axial':
                min_y, max_y = r1, r2
                min_x, max_x = c1, c2
                min_z, max_z = sel_slice_idx, sel_slice_idx
            elif sel_plane == 'coronal':
                min_z, max_z = r1, r2
                min_x, max_x = c1, c2
                min_y, max_y = sel_slice_idx, sel_slice_idx
            elif sel_plane == 'sagital':
                min_z, max_z = r1, r2
                min_y, max_y = c1, c2
                min_x, max_x = sel_slice_idx, sel_slice_idx

            if min_z <= slice_to_display_axial <= max_z: # Check if current axial slice intersects the 3D ROI
                rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                     fill=False, edgecolor='cyan', linewidth=1, linestyle='--')
                ax2.add_patch(rect)

        plt.tight_layout()
        plt.show(block=False)
        
        return fig
    
    def refine_selection(self):
        """Permite refinar la selección después de ver el resultado, con navegación 3D y selección de plano."""
        if self.segmented_mask is None:
            messagebox.showwarning("Advertencia", "Primero debes realizar una segmentación para refinarla.")
            return None
        
        if self.fig_selection is not None and plt.fignum_exists(self.fig_selection.number):
            plt.close(self.fig_selection)

        original_array_sitk = self.dicom_data
        image_array_zyx = sitk.GetArrayFromImage(original_array_sitk) # Z, Y, X
        
        num_slices_z, height_y, width_x = image_array_zyx.shape
        
        # Use the axial slice of the segmentation to start refinement
        slice_to_display_axial = self.current_slice_axial

        self.fig_selection, axs = plt.subplots(1, 3, figsize=(18, 6))
        self.ax_axial, self.ax_coronal, self.ax_sagital = axs
        
        # --- Vista Axial (Z) ---
        self.img_axial = self.ax_axial.imshow(image_array_zyx[slice_to_display_axial, :, :], cmap='gray')
        self.ax_axial.imshow(self.segmented_mask[slice_to_display_axial, :, :], cmap='Reds', alpha=0.7)
        self.ax_axial.set_title(f'Axial (Slice {slice_to_display_axial}/{num_slices_z-1}) - Refine here')
        self.ax_axial.axis('off')
        
        # --- Vista Coronal (Y) ---
        self.img_coronal = self.ax_coronal.imshow(image_array_zyx[:, self.current_slice_coronal, :], cmap='gray')
        self.ax_coronal.imshow(self.segmented_mask[:, self.current_slice_coronal, :], cmap='Reds', alpha=0.7)
        self.ax_coronal.set_title(f'Coronal (Slice {self.current_slice_coronal}/{height_y-1})')
        self.ax_coronal.axis('off')
        
        # --- Vista Sagital (X) ---
        self.img_sagital = self.ax_sagital.imshow(image_array_zyx[:, :, self.current_slice_sagital], cmap='gray')
        self.ax_sagital.imshow(self.segmented_mask[:, :, self.current_slice_sagital], cmap='Reds', alpha=0.7)
        self.ax_sagital.set_title(f'Sagital (Slice {self.current_slice_sagital}/{width_x-1})')
        self.ax_sagital.axis('off')

        plt.tight_layout()

        # Sliders (same as in select_region_first, but for refinement)
        axcolor = 'lightgoldenrodyellow'
        ax_slider_axial = plt.axes([0.1, 0.05, 0.2, 0.03], facecolor=axcolor)
        self.slider_axial = Slider(ax_slider_axial, 'Axial Slice', 0, num_slices_z - 1, valinit=slice_to_display_axial, valstep=1)
        self.slider_axial.on_changed(self._update_axial_slice)

        ax_slider_coronal = plt.axes([0.4, 0.05, 0.2, 0.03], facecolor=axcolor)
        self.slider_coronal = Slider(ax_slider_coronal, 'Coronal Slice', 0, height_y - 1, valinit=self.current_slice_coronal, valstep=1)
        self.slider_coronal.on_changed(self._update_coronal_slice)
        
        ax_slider_sagital = plt.axes([0.7, 0.05, 0.2, 0.03], facecolor=axcolor)
        self.slider_sagital = Slider(ax_slider_sagital, 'Sagital Slice', 0, width_x - 1, valinit=self.current_slice_sagital, valstep=1)
        self.slider_sagital.on_changed(self._update_sagital_slice)

        # --- Buttons to select the selection plane for refinement ---
        ax_btn_axial = plt.axes([0.1, 0.005, 0.08, 0.035])
        btn_axial = Button(ax_btn_axial, 'Select Axial')
        btn_axial.on_clicked(lambda event: self._activate_selector_on_plane('axial')) # Using the same selector activation logic

        ax_btn_coronal = plt.axes([0.4, 0.005, 0.08, 0.035])
        btn_coronal = Button(ax_btn_coronal, 'Select Coronal')
        btn_coronal.on_clicked(lambda event: self._activate_selector_on_plane('coronal'))
        
        ax_btn_sagital = plt.axes([0.7, 0.005, 0.08, 0.035])
        btn_sagital = Button(ax_btn_sagital, 'Select Sagital')
        btn_sagital.on_clicked(lambda event: self._activate_selector_on_plane('sagital'))

        # Initialize the selector on the axial plane by default for refinement
        self._activate_selector_on_plane('axial') 
        
        self.fig_selection.canvas.mpl_connect('close_event', self.on_close_selection_fig)
        plt.show(block=False)
        
        return self.fig_selection
    
    def on_refine_selection(self, eclick, erelease):
        """Callback para refinamiento de selección. Actualiza la región 3D."""
        # This function is currently not explicitly connected, as _activate_selector_on_plane
        # uses on_select_region internally. If you want a different callback for refinement,
        # you would need to change how RectangleSelector is initialized in refine_selection.
        # For now, it will use on_select_region which updates selected_region.
        self.on_select_region(eclick, erelease) # Reuse the existing selection logic
        print("Región de refinamiento actualizada.")

    def extract_final_bone(self):
        """
        Extrae el mayor componente conectado de la máscara segmentada.
        Esto es útil si la segmentación generó múltiples componentes y quieres aislar el hueso principal.
        """
        if self.segmented_mask is None or np.sum(self.segmented_mask) == 0:
            print("No hay máscara segmentada o está vacía para extraer el hueso final.")
            messagebox.showwarning("Advertencia", "No hay máscara segmentada o está vacía para extraer el hueso final.")
            return None
        
        # Apply Connected Component Analysis
        labels_out, num_labels = ndimage.label(self.segmented_mask, structure=np.ones((3,3,3))) # Use 3D connectivity
        
        if num_labels == 0:
            print("No se encontraron componentes conectados en la máscara segmentada.")
            messagebox.showinfo("Componentes Conectados", "No se encontraron componentes conectados significativos.")
            return None
        
        if num_labels == 1:
            print("Un solo componente conectado encontrado, usando la máscara tal cual.")
            messagebox.showinfo("Componentes Conectados", "Un solo componente conectado encontrado, usando la máscara tal cual.")
            return self.segmented_mask
            
        # Encontrar el componente más grande
        # Ensure we are summing over the original binary mask for accurate size
        component_sizes = ndimage.sum_labels(self.segmented_mask, labels_out, index=range(1, num_labels + 1))
        
        if len(component_sizes) == 0:
            print("No se encontraron componentes conectados significativos.")
            return None

        # largest_label is 1-indexed
        largest_label_idx = np.argmax(component_sizes) + 1 
        
        final_mask = (labels_out == largest_label_idx).astype(np.uint8)
        print(f"Extraído el componente más grande (label {largest_label_idx}). Voxels: {np.sum(final_mask)}")
        messagebox.showinfo("Extracción de Hueso Principal", f"Extraído el componente más grande. Voxels: {np.sum(final_mask)}")
        return final_mask
    
    def create_stl(self, mask, output_path, make_hollow=False):
        """Convierte la máscara en un archivo STL"""
        if mask is None:
            messagebox.showwarning("Advertencia", "No hay máscara para crear el STL.")
            return False
        
        try:
            print(f"Máscara original para STL: {mask.shape}, voxels: {np.sum(mask)}")
            
            # If it's a single slice (2D), convert to 3D by extruding
            if mask.ndim == 2: # Check if it's 2D
                print("Máscara 2D detectada. Extruyendo a 3D para STL...")
                thickness = 5 # Adjust this value as needed for "depth"
                # Add a new axis at the beginning (Z axis)
                new_mask = np.zeros((thickness, mask.shape[0], mask.shape[1]), dtype=mask.dtype)
                for i in range(thickness):
                    new_mask[i] = mask
                mask = new_mask
                print(f"Nueva forma extruida: {mask.shape}")
            
            # Ensure minimum size for marching_cubes (at least 2x2x2)
            min_dim_for_marching_cubes = 2 
            
            if any(dim < min_dim_for_marching_cubes for dim in mask.shape):
                print(f"Redimensionando máscara a tamaño mínimo para Marching Cubes ({min_dim_for_marching_cubes}x{min_dim_for_marching_cubes}x{min_dim_for_marching_cubes})...")
                original_shape = mask.shape
                new_shape = [max(dim, min_dim_for_marching_cubes) for dim in original_shape]
                temp_mask = np.zeros(new_shape, dtype=mask.dtype)
                
                # Copy the original data into the top-left-front corner of the new larger mask
                temp_mask[:original_shape[0], :original_shape[1], :original_shape[2]] = mask
                mask = temp_mask
                print(f"Máscara redimensionada a: {mask.shape}")

            # Apply smoothing if necessary (before Marching Cubes) for very small or thin objects
            if np.sum(mask) > 0 and np.sum(mask) < 200: # Only if there's something and it's very small
                print("Aplicando dilatación para asegurar volumen adecuado antes de Marching Cubes...")
                mask = morphology.binary_dilation(mask, morphology.ball(1)).astype(np.uint8) 
            
            print(f"Generando mesh con forma: {mask.shape}, voxels: {np.sum(mask)}")
            
            # Generate mesh
            if np.sum(mask) == 0:
                print("La máscara está vacía, no se puede generar STL.")
                messagebox.showerror("Error STL", "La máscara está vacía, no se puede generar STL.")
                return False

            if np.max(mask) == np.min(mask):
                print("La máscara no tiene variación de valores (todo 0 o todo 1), no se puede generar STL.")
                messagebox.showerror("Error STL", "La máscara no tiene variación de valores (todo 0 o todo 1), no se puede generar STL.")
                return False
            
            # Add padding to avoid issues with marching_cubes at image boundaries
            padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
            
            verts, faces, _, _ = measure.marching_cubes(
                padded_mask, # Use padded mask
                level=0.5, 
                spacing=(1.0, 1.0, 1.0), 
                allow_degenerate=False
            )
            
            print(f"Mesh generado: {len(verts)} vértices, {len(faces)} caras")
            
            # Make hollow structure if requested
            if make_hollow and mask.shape[0] > 6: # Ensure enough depth for erosion
                print("Intentando crear estructura hueca...")
                # Erosion needs to be applied to the original mask, not the padded one for consistent result
                eroded_mask = morphology.binary_erosion(mask, morphology.ball(2)) 
                
                # Pad eroded mask before marching cubes
                padded_eroded_mask = np.pad(eroded_mask, pad_width=1, mode='constant', constant_values=0)

                if np.sum(padded_eroded_mask) > 100: # Ensure eroded mask is not too small
                    try:
                        inner_verts, inner_faces, _, _ = measure.marching_cubes(
                            padded_eroded_mask, level=0.5, spacing=(1.0, 1.0, 1.0),
                            allow_degenerate=False
                        )
                        # Flip normals for inner surface
                        inner_faces = inner_faces[:, ::-1] 
                        
                        all_verts = np.vstack([verts, inner_verts])
                        all_faces = np.vstack([faces, inner_faces + len(verts)])
                        verts, faces = all_verts, all_faces
                        print("Estructura hueca creada exitosamente")
                    except Exception as e:
                        print(f"No se pudo crear estructura hueca completamente (posiblemente muy pequeño): {e}, usando sólida")
                        messagebox.showwarning("Hueco Fallido", f"No se pudo crear estructura hueca completamente: {e}. Usando sólida.")
                else:
                    print("La máscara erosionada es demasiado pequeña o vacía para crear un hueco significativo, usando sólida.")
                    messagebox.showwarning("Hueco Fallido", "La máscara erosionada es demasiado pequeña para crear un hueco significativo. Usando sólida.")
            
            # Write STL file
            if len(faces) == 0: 
                print("No se generaron caras para el STL. La máscara podría ser demasiado pequeña o no tener una superficie bien definida.")
                messagebox.showerror("Error STL", "No se generaron caras para el STL. La máscara podría ser demasiado pequeña o no tener una superficie bien definida.")
                return False

            self._write_stl(verts, faces, output_path)
            print(f"Archivo STL guardado: {output_path}")
            messagebox.showinfo("Exportar STL", f"Archivo STL guardado exitosamente: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creando STL: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error STL", f"Ocurrió un error al crear el archivo STL: {e}")
            return False
    
    def _write_stl(self, vertices, faces, filename):
        """Escribe un archivo STL ASCII"""
        with open(filename, 'w') as f:
            f.write('solid bone\n')
            
            for face in faces:
                v1, v2, v3 = vertices[face]
                # Calculate normal vector
                normal = np.cross(v2 - v1, v3 - v1)
                norm_length = np.linalg.norm(normal)
                if norm_length > 0:
                    normal = normal / norm_length
                else:
                    normal = np.array([0, 0, 1]) # Default if normal is zero
                
                f.write(f'facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n')
                f.write('outer loop\n')
                for vertex_idx in face:
                    v = vertices[vertex_idx]
                    f.write(f'vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
                f.write('endloop\n')
                f.write('endfacet\n')
            
            f.write('endsolid bone\n')

class BoneSegmenterGUI:
    def __init__(self):
        self.segmenter = BoneSegmenter()
        self.root = tk.Tk()
        self.root.title("Segmentador de Huesos DICOM - Optimizado")
        self.root.geometry("700x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_gui()
    
    def on_closing(self):
        # Ensure all matplotlib figures are closed when the Tkinter GUI is closed
        plt.close('all') 
        self.root.destroy()

    def setup_gui(self):
        # Frame for file operations
        file_frame = ttk.LabelFrame(self.root, text="Operaciones de Archivo")
        file_frame.pack(padx=10, pady=5, fill="x")

        ttk.Button(file_frame, text="Cargar Serie DICOM", command=self.load_dicom_series_folder).pack(side="left", padx=5, pady=5)
        ttk.Button(file_frame, text="Cargar DICOM Individual", command=self.load_single_dicom_file).pack(side="left", padx=5, pady=5)
        
        # Frame for segmentation parameters
        param_frame = ttk.LabelFrame(self.root, text="Parámetros de Segmentación")
        param_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(param_frame, text="Min HU:").pack(side="left", padx=5, pady=5)
        self.min_hu_entry = ttk.Entry(param_frame, width=10)
        self.min_hu_entry.insert(0, str(self.segmenter.hu_threshold_min))
        self.min_hu_entry.pack(side="left", padx=5, pady=5)

        ttk.Label(param_frame, text="Max HU:").pack(side="left", padx=5, pady=5)
        self.max_hu_entry = ttk.Entry(param_frame, width=10)
        self.max_hu_entry.insert(0, str(self.segmenter.hu_threshold_max))
        self.max_hu_entry.pack(side="left", padx=5, pady=5)

        # Frame for actions
        action_frame = ttk.LabelFrame(self.root, text="Acciones")
        action_frame.pack(padx=10, pady=5, fill="x")

        ttk.Button(action_frame, text="Seleccionar Región (3D)", command=self.select_region).pack(side="left", padx=5, pady=5)
        ttk.Button(action_frame, text="Segmentar Región Seleccionada", command=self.segment_selected_region).pack(side="left", padx=5, pady=5)
        ttk.Button(action_frame, text="Segmentar Imagen Completa", command=self.segment_full_image).pack(side="left", padx=5, pady=5)
        ttk.Button(action_frame, text="Mostrar Resultado", command=self.show_segmentation).pack(side="left", padx=5, pady=5)
        ttk.Button(action_frame, text="Refinar Selección", command=self.refine_segmentation).pack(side="left", padx=5, pady=5)
        ttk.Button(action_frame, text="Extraer Hueso Principal", command=self.extract_main_bone).pack(side="left", padx=5, pady=5)
        
        # Frame for STL export
        stl_frame = ttk.LabelFrame(self.root, text="Exportar STL")
        stl_frame.pack(padx=10, pady=5, fill="x")
        
        self.make_hollow_var = tk.BooleanVar()
        ttk.Checkbutton(stl_frame, text="Hacer Hueco", variable=self.make_hollow_var).pack(side="left", padx=5, pady=5)
        ttk.Button(stl_frame, text="Guardar STL de Hueso Principal", command=self.save_main_bone_stl).pack(side="left", padx=5, pady=5)
        ttk.Button(stl_frame, text="Guardar STL de Máscara Completa", command=self.save_full_mask_stl).pack(side="left", padx=5, pady=5)

    def load_dicom_series_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            if self.segmenter.load_dicom_series(folder_selected):
                messagebox.showinfo("Carga Exitosa", "Serie DICOM cargada correctamente.")
            else:
                messagebox.showerror("Error de Carga", "No se pudo cargar la serie DICOM. Asegúrate de que la carpeta contiene archivos DICOM válidos.")

    def load_single_dicom_file(self):
        file_selected = filedialog.askopenfilename(filetypes=[("DICOM files", ".dcm *.DCM"), ("All files", ".*")])
        if file_selected:
            if self.segmenter.load_single_dicom(file_selected):
                messagebox.showinfo("Carga Exitosa", "Archivo DICOM individual cargado correctamente.")
            else:
                messagebox.showerror("Error de Carga", "No se pudo cargar el archivo DICOM individual.")

    def select_region(self):
        self.segmenter.select_region_first()

    def segment_selected_region(self):
        try:
            min_hu = int(self.min_hu_entry.get())
            max_hu = int(self.max_hu_entry.get())
            self.segmenter.hu_threshold_min = min_hu
            self.segmenter.hu_threshold_max = max_hu
        except ValueError:
            messagebox.showerror("Error de Entrada", "Los valores de HU deben ser números enteros.")
            return

        if self.segmenter.segment_bones_in_region(min_hu, max_hu):
            # The success/failure message is handled within segment_bones_in_region
            pass # No need for additional messagebox here

    def segment_full_image(self):
        try:
            min_hu = int(self.min_hu_entry.get())
            max_hu = int(self.max_hu_entry.get())
            self.segmenter.hu_threshold_min = min_hu
            self.segmenter.hu_threshold_max = max_hu
        except ValueError:
            messagebox.showerror("Error de Entrada", "Los valores de HU deben ser números enteros.")
            return

        if self.segmenter.segment_bones_full(min_hu, max_hu):
            # The success/failure message is handled within segment_bones_full
            pass # No need for additional messagebox here

    def show_segmentation(self):
        self.segmenter.show_segmentation_result()

    def refine_segmentation(self):
        self.segmenter.refine_selection()

    def extract_main_bone(self):
        if self.segmenter.segmented_mask is None:
            messagebox.showwarning("Advertencia", "No hay segmentación para extraer el hueso principal.")
            return
        
        final_mask = self.segmenter.extract_final_bone()
        if final_mask is not None:
            self.segmenter.segmented_mask = final_mask # Update the main mask with the extracted bone

    def save_main_bone_stl(self):
        if self.segmenter.segmented_mask is None:
            messagebox.showwarning("Advertencia", "No hay máscara segmentada para guardar como STL.")
            return
        
        final_mask = self.segmenter.extract_final_bone() # Get the largest connected component
        if final_mask is None:
            messagebox.showwarning("Advertencia", "No se pudo extraer un hueso principal para guardar.")
            return

        output_file = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL files", "*.stl")])
        if output_file:
            make_hollow = self.make_hollow_var.get()
            self.segmenter.create_stl(final_mask, output_file, make_hollow)

    def save_full_mask_stl(self):
        if self.segmenter.segmented_mask is None:
            messagebox.showwarning("Advertencia", "No hay máscara segmentada para guardar como STL.")
            return
        
        output_file = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL files", "*.stl")])
        if output_file:
            make_hollow = self.make_hollow_var.get()
            self.segmenter.create_stl(self.segmenter.segmented_mask, output_file, make_hollow)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = BoneSegmenterGUI()
    app.run()