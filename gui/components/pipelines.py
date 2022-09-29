from __future__ import annotations
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Type, List
import logging
import pathlib
import tkinter as tk
from tkinter import ttk

from . import sections 


class _Scrollable(tk.Frame):
    """
       Make a frame scrollable with scrollbar on the right.
       After adding or removing widgets to the scrollable frame,
       call the update() method to refresh the scrollable area.
    """

    def __init__(self, frame, width=16, *args, **kwargs):

        scrollbar = tk.Scrollbar(frame, width=width)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, expand=False)

        self.canvas = tk.Canvas(frame, yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.canvas.yview)

        self.canvas.bind('<Configure>', self.__fill_canvas)

        # base class initialization
        tk.Frame.__init__(self, frame, *args, **kwargs)

        # assign this obj (the inner frame) to the windows item of the canvas
        self.windows_item = self.canvas.create_window(0,0, window=self, anchor=tk.NW)

    def __fill_canvas(self, event):
        "Enlarge the windows item to the canvas width"

        canvas_width = event.width
        self.canvas.itemconfig(self.windows_item, width = canvas_width)

    def update(self):
        "Update the canvas and the scrollregion"

        self.update_idletasks()


class PipelineOptions(tk.Frame, ABC):
    @staticmethod
    def _get_children_of_type(master: Type[PipelineOptions],
                              widget_type: Type) -> List:
        children = master.winfo_children()
        
        return [c for c in children if isinstance(c, widget_type)]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sections = OrderedDict()
        self.build()
    
    @property
    def frames(self):
        return self._get_children_of_type(self, tk.Frame)
 
    @property
    def labels(self):
        return self._get_children_of_type(self, tk.Label)
    
    @property
    def comboboxes(self):
        return self._get_children_of_type(self, ttk.Combobox)
            
    @abstractmethod
    def build(self):
        ...
 
    @abstractmethod
    def run(self):
        ...
    
    @abstractmethod
    def save(self):
        ...   
    

class BlankPipelineOptions(PipelineOptions):
    def build(self):
        ...

    def run(self):
        ...

    def save(self):
        ...


class EoRH21cmOptions(PipelineOptions):
    def build(self):        
        self.sections['output'] = sections.output(self)
        self.sections['field'] = sections.field(self)
        self.sections['correlator'] = sections.correlator(self)
        self.sections['astro_params'] = sections.eor_astro_params(self)
        self.sections['user_params'] = sections.eor_user_params(self)
        self.sections['flags'] = sections.eor_flags(self)

        for section in self.sections.values():
            section.build()
            section.pack(fill='x')

        # TODO: Find place for the below code within sections.field function and
        #  EntryField class
        def update_cdelt(event: tk.Event):
            try:
                new_val = (self.sections['field'].fields['fov'].value /
                           self.sections['field'].fields['n_cell'].value)

                self.sections['field'].fields['cdelt'].var.set(new_val)
            except ZeroDivisionError:
                self.sections['field'].fields['cdelt'].var.set(0.)

        def update_ncell(event: tk.Event):
            try:
                new_val = int(self.sections['field'].fields['fov'].value /
                              self.sections['field'].fields['cdelt'].value)
                self.sections['field'].fields['n_cell'].var.set(new_val)
            except ZeroDivisionError:
                pass
            finally:
                update_cdelt(event)

        self.sections['field'].fields['cdelt'].entry.bind("<FocusOut>",
                                                          update_ncell)

        self.sections['field'].fields['n_cell'].entry.bind("<FocusOut>",
                                                           update_cdelt)

    def run(self):
        from ...physics.eor_h21cm import create_eor_h21cm_fits

        params_toml = self.save(ask_save_file=False)
        logging.log(logging.INFO, "Running EoR H21cm code")

        create_eor_h21cm_fits(params_toml)

    def save(self, ask_save_file: bool = True):
        import toml
        from .sections import _CustomDialog

        d = {k: v.to_dict() for k, v in self.sections.items()}

        save_dcy = pathlib.Path(d['output']['output_dcy'])
        
        if not save_dcy.exists():
            save_dcy.mkdir()
        
        save_file = save_dcy / f"{d['output']['root_name']}.toml"

        if ask_save_file:
            dialog = _CustomDialog(self, "Save path:", save_file)
            save_file = dialog.show()
        
            if save_file is None:
                return None

        with open(save_file, 'wt') as f:
            toml.dump(d, f)
        
        return save_file


_AVAILABLE_PIPELINES = ["EoR H21cm", "TRECS", "MHD", "SDC3"]
_PIPELINE_FRAMES = {
            'EoR H21cm': EoRH21cmOptions,
            'TRECS': BlankPipelineOptions,
            'MHD': BlankPipelineOptions,
            'SDC3': BlankPipelineOptions,
        }
