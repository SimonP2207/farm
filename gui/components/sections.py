from __future__ import annotations
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Type, Any, Optional
from collections import OrderedDict
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from tkinter import font

try:
    import sympy as sp
    _SYMPY_INSTALLED = True
except ModuleNotFoundError:
    _SYMPY_INSTALLED = False


class _CustomDialog(tk.Toplevel):
    def __init__(self, parent: tk.Frame, prompt: str,
                 default: Optional[Any] = None):
        tk.Toplevel.__init__(self, parent)
        self.geometry('480x80')

        self.var = tk.StringVar(value=default)

        self.label = tk.Label(self, text=prompt)
        self.entry = tk.Entry(self, textvariable=self.var)
        self.ok_button = tk.Button(self, text="Save", command=self.on_save,
                                   width=100)
        self.cx_button = tk.Button(self, text="Cancel", command=self.on_cancel,
                                   width=100)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
                
        self.label.grid(row=0, column=0, sticky='nwes')
        self.entry.grid(row=0, column=1, columnspan=3, sticky='ew')
        
        tk.Label(self, text='', width=100).grid(row=1, column=0)  # spacing
        tk.Label(self, text='', width=100).grid(row=1, column=1)  # spacing
        self.ok_button.grid(row=1, column=2)
        self.cx_button.grid(row=1, column=3)

        self.entry.bind("<Return>", self.on_save)
        self.entry.bind("<Escape>", self.on_cancel)
        
        self.save_clicked = False

    def on_save(self, event=None):
        self.save_clicked = True
        self.destroy()
        
    def on_cancel(self, event=None):
        self.destroy()

    def show(self):
        self.wm_deiconify()
        self.entry.focus_force()
        self.wait_window()

        if self.save_clicked:
            return self.var.get()


class _ToggledFrame(tk.Frame):
    def __init__(self, text: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.show = tk.IntVar()
        self.show.set(0)

        self.title_frame = ttk.Frame(self, relief='groove', border=1)
        self.title_frame.grid(sticky='nsew')
        self.title_frame.columnconfigure(0, weight=1)
        self.title_frame.columnconfigure(0, weight=1)
        self.title_frame.rowconfigure(0, weight=1)

        self.label = ttk.Label(self.title_frame, text=text)
        
        default_font = font.nametofont("TkDefaultFont").actual()
        
        self.label.configure(font=(default_font['family'],
                                   default_font['size'],
                                   "bold"))

        self.toggle_button = ttk.Checkbutton(
            self.title_frame, width=2, text=u'\u25B2', command=self.toggle,
            variable=self.show, style='Toolbutton',
            )
        
        self.label.grid(column=0, row=0, sticky="nsew")
        self.toggle_button.grid(column=1, row=0, sticky="nsew")

        self.content_frame = tk.Frame(self, relief="flat", borderwidth=1,
                                      padx=20, pady=0)

        self.toggle()

    def toggle(self):
        if bool(self.show.get()):
            self.content_frame.grid(row=1, column=0, sticky='nsew')
            self.toggle_button.configure(text=u'\u25BC')
        else:
            self.content_frame.grid_remove()
            self.toggle_button.configure(text=u'\u25B2')


class Section(_ToggledFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields = OrderedDict()
        self.buttons = []

    def build(self):        
        for field in self.fields.values():
            field.build()
            field.pack(fill='x')
            # field.grid(row=row + 1, column=0, sticky='nsew')
        
        for button in self.buttons:
            button.pack(side=tk.RIGHT)

    def to_dict(self):
        return {k: v.value for k, v in self.fields.items()}

class Field(tk.Frame, ABC):
    """        
    Abstract class for a single field in a section. Contains a label, and data 
    widget (e.g. Entry, CheckBox)
    """
    def __init__(self, text: str, hint: str, var_type: Type,
                 default: Optional[Any] = None, latex: bool = False,
                 *args, **kwargs):
        """
        Parameters
        ----------
        text : str
            Label's text
        hint : str
            Text to show as a hover box when mouse hovers over data widget
        var_type : Type
            Type of variable e.g. float, str
        default : Optional[Any], optional
            Default value in textbox, by default None
        latex : bool, optional
            Whether to render 'text' argument as LaTeX math, by default False
        """
        super().__init__(bg='#ececec', *args, **kwargs)

        self.text = text
        self.hint = hint
        self.var_type = var_type
        self.default = default #if default is not None else ""
        self.latex = latex
              
        self.label = None  # Text label
        self.entry = None  # Entry box etc.
        self.hover_tip = None  # Box to display with hint upon mouse hover
        
        tk_vars = {int: tk.IntVar, str: tk.StringVar,
                   bool: tk.BooleanVar, float: tk.DoubleVar}

        self.var = tk_vars[self.var_type](value=self.default)
    
    @property
    def value(self):
        return self.var.get()
    
    def configure_grid(self):
        """Configure the Field's columns and rows"""
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        
    @abstractmethod
    def create_data_entry(self):
        ...
    
    @abstractmethod
    def grid_label_and_data_entry(self):
        ...
    
    def build(self):
        """Build the Field by configuring the Field's grid (columns and rows), rendering text to LaTeX if requested, assigning the """
        from idlelib.tooltip import Hovertip

        self.configure_grid()
        
        if self.latex:
            self.label = ttk.Label(self)
            math_expr = self._mathmode(self.text)
            self.label.config(image=math_expr)
            self.label.image = math_expr
        else:
            self.label = ttk.Label(self, text=self.text)
        
        self.entry = self.create_data_entry()
        self.grid_label_and_data_entry()
        self.hover_tip = Hovertip(self.entry, self.hint, hover_delay=500)

    def _mathmode(self, text: str) -> ImageTk.PhotoImage:
        """
        Convert text to LaTeX mathmode

        Parameters
        ----------
        text : str
            Text to convert to LaTeX mathmode

        Returns
        -------
        ImageTk.PhotoImage
            LaTeX mathmode image
        """
        from io import BytesIO

        
        # Create a ByteIO stream instance to save the output of sympy.preview to
        f = BytesIO()
        bg_color = f"{{{self['background'].upper()[1:]}}}"

        sp.preview(text, euler = False, 
                   preamble = r"\documentclass{standalone}"
                              r"\usepackage{pagecolor}"
                              r"\definecolor{bgcolor}{HTML}"f"{bg_color}"
                              r"\usepackage{cmbright}"
                              r"\usepackage[OT1]{fontenc}"
                              r"\pagecolor{bgcolor}"
                              r"\begin{document}",
                   viewer="BytesIO", output="ps", outputbuffer=f)
        f.seek(0)
        
        # Open the image as if it were a file. This works only for .ps
        img = Image.open(f)
        img.load(scale=20)

        img = img.resize((int(img.size[0] / 16), int(img.size[1] / 16)),
                         Image.BICUBIC)

        photo = ImageTk.PhotoImage(img)

        f.close()
        
        return photo


class EntryField(Field):
    def create_data_entry(self):
        return ttk.Entry(self, name=self.text.lower(), textvariable=self.var)

    def grid_label_and_data_entry(self):
        self.label.grid(row=0, column=0, sticky='nsw')
        self.entry.grid(row=0, column=1, sticky='nse')


class CheckbuttonField(Field):
    def __init__(self, text: str, hint: str, default: Optional[bool] = True,
                 latex: bool = False, *args, **kwargs):
        """
        Parameters
        ----------
        text : str
            Label's text
        hint : str
            Text to show as a hover box when mouse hovers over data widget
        default : Optional[Any], optional
            Default value in textbox, by default None
        latex : bool, optional
            Whether to render 'text' argument as LaTeX math, by default False
        """
        super().__init__(text, hint, bool, default, latex, *args, **kwargs)

    def create_data_entry(self):
        return ttk.Checkbutton(self, name=self.text.lower(), variable=self.var)

    def grid_label_and_data_entry(self):
        self.label.grid(row=0, column=0, sticky='nsw')
        self.entry.grid(row=0, column=1, sticky='ens')
        

# Predefined, common Sections below
def output(master: Type[tk.Frame]) -> Section:
    """
    Section instance for output and common fields for such

    Parameters
    ----------
    master : Type[tk.Frame]
        Master frame for the output section

    Returns
    -------
    Section
        Section instance with common output fields
    """
    output = Section(master=master, text="Output Options")
    output.fields['output_dcy'] = EntryField(
        master=output.content_frame, text="Output Directory",
        hint="Directory to save all products", var_type=str,
        default=Path('~').expanduser()
    )
    
    output.fields['root_name'] = EntryField(
        master=output.content_frame, text="Output Name",
        hint="Full path to save EoR data cube to", var_type=str,
        default="EoR_H21cm"
    )

    return output


def field(master: Type[tk.Frame]) -> Section:
    """
    Section instance for astronomical field details and common fields for such

    Parameters
    ----------
    master : Type[tk.Frame]
        master frame for the astronomical field section

    Returns
    -------
    Section
        Section instance with common output fields
    """
    field = Section(master=master, text="Field Details")
    
    field.fields['ra0'] = EntryField(
        master=field.content_frame, text="Right Ascension (J2000) [deg]",
        hint="Pointing centre right ascension in degrees", 
        var_type=float, default=0.
    )

    field.fields['dec0'] = EntryField(
        master=field.content_frame, text="Declination (J2000) [deg]",
        hint="Pointing centre declination in degrees", 
        var_type=float, default=-30.
    )

    field.fields['fov'] = EntryField(
        master=field.content_frame, text="Field of View [deg]",
        hint="Field of view in both RA/declination in degrees",
        var_type=float, default=8.
    )

    field.fields['cdelt'] = EntryField(
        master=field.content_frame, text="Cell size [deg]",
        hint="Pixel size in RA/declination in degrees",
        var_type=float, default=float(format(8. / 1024, '.6f'))
    )

    return field


def correlator(master: Type[tk.Frame]) -> Section:
    """
    Section instance for correlator details and common fields for such

    Parameters
    ----------
    master : Type[tk.Frame]
        master frame for the correlator section

    Returns
    -------
    Section
        Section instance with common output fields
    """
    correlator = Section(master=master, text="Correlator Setup")
    
    correlator.fields['freq_min'] = EntryField(
        master=correlator.content_frame,
        text=r"$\nu_\mathrm{min}\,\left[ \mathrm{Hz} \right]$"
        if _SYMPY_INSTALLED else 'Minimum frequency [Hz]',
        hint="Minimum frequency in Hz", 
        var_type=float, latex=_SYMPY_INSTALLED, default=100e6,
    )

    correlator.fields['freq_max'] = EntryField(
        master=correlator.content_frame,
        text=r"$\nu_\mathrm{max}\,\left[ \mathrm{Hz} \right]$"
        if _SYMPY_INSTALLED else 'Maximum frequency [Hz]',
        hint="Maximum frequency in Hz", 
        var_type=float, latex=_SYMPY_INSTALLED, default=200e6
    )

    correlator.fields['nchan'] = EntryField(
        master=correlator.content_frame,
        text=r"$n_\mathrm{chan}$" if _SYMPY_INSTALLED else 'Number of channels',
        hint="Number of frequency channels evenly spaced across bandwidth in Hz",
        var_type=int, latex=_SYMPY_INSTALLED, default=101
    )

    return correlator


def eor_astro_params(master: Type[tk.Frame]) -> Section:
    """
    Section instance for correlator details and common fields for such

    Parameters
    ----------
    master : Type[tk.Frame]
        master frame for the correlator section

    Returns
    -------
    Section
        Section instance with common output fields
    """
    params = Section(master=master, text="EoR Astrophysical Parameters")
    
    params.fields['F_STAR10'] = EntryField(
        master=params.content_frame,
        text=r"$\log_{10} \left(F_{\star,10}\right)$" if _SYMPY_INSTALLED else 'F_STAR10',
        hint="Log10 of the fraction of galactic gas in stars for "
             "10^10 solar mass haloes. Should be between -3.0 to 0.0", 
        var_type=float, latex=_SYMPY_INSTALLED, default=-1.3
    )

    params.fields['ALPHA_STAR'] = EntryField(
        master=params.content_frame,
        text=r"$\alpha_\star$" if _SYMPY_INSTALLED else 'ALPHA_STAR',
        hint="Power-law index of fraction of galactic gas in stars as "
             "a function of halo mass. Should be between -0.5 and 1.0", 
        var_type=float, latex=_SYMPY_INSTALLED, default=0.5
    )

    params.fields['F_ESC10'] = EntryField(
        master=params.content_frame, 
        text=r"$\log_{10} \left(F_\mathrm{esc,10}\right)$" if _SYMPY_INSTALLED else 'F_ESC10',
        hint="Log10 of the 'escape fraction', i.e. the fraction of "
             "ionizing photons escaping into the IGM, for 10^10 solar "
             "mass haloes. Should be between -3.0 and 0.0", 
        var_type=float, latex=_SYMPY_INSTALLED, default=-1.0
    )

    params.fields['ALPHA_ESC'] = EntryField(
        master=params.content_frame,
        text=r"$\alpha_\mathrm{esc}$" if _SYMPY_INSTALLED else 'ALPHA_ESC',
        hint="Power-law index of escape fraction as a function of halo "
             "mass. Should be between -1.0 and 0.5", 
        var_type=float, latex=_SYMPY_INSTALLED, default=-0.3
    )

    params.fields['M_TURN'] = EntryField(
        master=params.content_frame, 
        text=r"$\log_{10} \left (M_\mathrm{turn} \, "
             r"\left[ {\mathrm{M_\odot}} \right] \right)$" if _SYMPY_INSTALLED else 'M_TURN',
        hint="Turnover mass (in log10 solar mass units) for quenching "
             "of star formation in halos, due to SNe or photo-heating "
             "feedback, or inefficient gas accretion. Should be "
             "between 8.0 and 10.0", 
        var_type=float, latex=_SYMPY_INSTALLED, default=8.9
    )

    params.fields['t_STAR'] = EntryField(
        master=params.content_frame,
        text=r"$t_\mathrm{\star}$" if _SYMPY_INSTALLED else 't_STAR',
        hint="Fractional characteristic time-scale (fraction of hubble "
             "time) defining the star-formation rate of galaxies. "
             "Should be between 0.01 and 1.0", 
        var_type=float, latex=_SYMPY_INSTALLED, default=0.38
    )

    params.fields['L_X'] = EntryField(
        master=params.content_frame,
        text=r"$\log_{10} \left(L_\mathrm{X}\right)$" if _SYMPY_INSTALLED else 'L_X',
        hint="Log10 of the specific X-ray luminosity per unit star "
             "formation escaping host galaxies. Should be between "
             "38.0 and 44.0",
        var_type=float, latex=_SYMPY_INSTALLED, default=40.
    )

    params.fields['NU_X_THRESH'] = EntryField(
        master=params.content_frame,
        text=r"$\nu_\mathrm{X,\,thresh} \, \left[ \mathrm{eV} \right]$" if _SYMPY_INSTALLED else 'NU_X_THRESH',
        hint="X-ray energy threshold for self-absorption by host "
             "galaxies in eV. Should be between 100 and 1500", 
        var_type=float, latex=_SYMPY_INSTALLED, default=500.
    )
    
    params.fields['X_RAY_SPEC_INDEX'] = EntryField(
        master=params.content_frame,
        text=r"$\gamma_\mathrm{X}$" if _SYMPY_INSTALLED else 'X_RAY_SPEC_INDEX',
        hint="X-ray spectral energy index. Should be between "
             "-1.0 and 3.0", 
        var_type=float, latex=_SYMPY_INSTALLED, default=1.
    )


    def get_possible_eor_params() -> pd.DataFrame:
        import json
        from .. import DATA_FILES
        
        with open(DATA_FILES["eor_params"], 'rb') as f:
            data = json.load(f)
        
        return pd.DataFrame(data[1:], columns=data[0]).sample().squeeze()


    def assign_eor_params():
        eor_params = get_possible_eor_params()
        
        for key, val in eor_params.items():
            if isinstance(key, tuple):
                key = key[0] 
            
            params.fields[key].var.set(float(format(val, '.3f')))

    params.buttons.append(tk.Button(params.content_frame, 
                                    text='Suggest Values',
                                    command=assign_eor_params))
 
    return params


def eor_user_params(master: Type[tk.Frame]) -> Section:
    """
    Section instance for specifying EoR user parameters and common fields for
    such

    Parameters
    ----------
    master : Type[tk.Frame]
        Master frame for the output section

    Returns
    -------
    Section
        Section instance with common EoR user parameters
    """
    import os 

    params = Section(master=master, text="User Parameters")

    params.fields['seed'] = EntryField(
        master=params.content_frame, text="Seed",
        var_type=int, default=12345, hint="Random number generator seed"
    )

    params.fields['n_cpu'] = EntryField(
        master=params.content_frame,
        text=r"$n_\mathrm{CPU}$" if _SYMPY_INSTALLED else 'Number of CPUs',
        var_type=int, latex=_SYMPY_INSTALLED, default=min([os.cpu_count(), 48]),
        hint="Number of CPUs to use. Default is 48 unless maximum available is "
             "below that. Beyond 48 threads, there are no appreciable speed-ups"
             " because of the scaling limitation"
    )
       
    params.fields['USE_FFTW_WISDOM'] = CheckbuttonField(
        master=params.content_frame, text="Use stored FFTW_WISDOMS?",
        hint="Whether or not to use stored FFTW_WISDOMs for improving "
             "performance of FFTs",
        default=True
    )
    
    params.fields['PERTURB_ON_HIGH_RES'] = CheckbuttonField(
        master=params.content_frame, text="Perform grid perturbation?",
        hint="Whether to perform the Zel'Dovich or 2LPT perturbation on the "
             "low or high resolution grid",
        default=True
    )

    params.fields['USE_INTERPOLATION_TABLES'] = CheckbuttonField(
        master=params.content_frame, text="Use Interpolation Tables?",
        hint="Whether to use interpolation tables. True makes the run faster",
        default=True
    )

    return params


def eor_flags(master: Type[tk.Frame]) -> Section:
    """
    Section instance for specifying EoR user parameters and common fields for
    such

    Parameters
    ----------
    master : Type[tk.Frame]
        Master frame for the output section

    Returns
    -------
    Section
        Section instance with common EoR user parameters
    """
    import os 

    eor_flags = Section(master=master, text="EoR Flags")

    eor_flags.fields['INHOMO_RECO'] = CheckbuttonField(
        master=eor_flags.content_frame,
        text="Compute inhomogeneous recombinations?",
        hint="Compute inhomogeneous recombinations?",
        default=True
    )
    
    eor_flags.fields['USE_MASS_DEPENDENT_ZETA'] = CheckbuttonField(
        master=eor_flags.content_frame,
        text="Scale Zeta with halo mass?",
        hint="Allow ionizing efficiency, zeta, to scale with the halo mass?",
        default=True
    )
 
    eor_flags.fields['USE_TS_FLUCT'] = CheckbuttonField(
        master=eor_flags.content_frame,
        text="Spin temperature fluctuations required?",
        hint="Whether to perform IGM spin temperature fluctuations (i.e. X-ray "
             "heating). Dramatically increases the computation time.",
        default=True
    )
    
    eor_flags.fields['USE_MINI_HALOS'] = CheckbuttonField(
        master=eor_flags.content_frame, text="Use mini-halos parameterisation?",
        hint="Whether to use updated radiation source model with:\n(i) all "
             "radiation fields including X-rays, UV ionizing, Lyman Werner and "
             "Lyman alpha are considered from two seperated population namely "
             "atomic-cooling (ACGs) and minihalo-hosted molecular-cooling "
             "galaxies (MCGs)\n(ii) the turn-over masses of ACGs and MCGs are "
             "estimated with cooling efficiency and feedback from reionization "
             "and lyman werner suppression (Qin et al. 2020)\n If True, must "
             "scale zeta with halo mass and compute inhomogeneous "
             "recombinations",
        default=False
    )

    eor_flags.fields['USE_PHOTON_CONS'] = CheckbuttonField(
        master=eor_flags.content_frame,
        text="Correct for photon non-conservation?",
        hint="Whether to perform a small correction to account for the "
             "inherent photon non-conservation",
        default=True
    )

    return eor_flags
