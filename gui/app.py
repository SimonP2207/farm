import tkinter as tk
from tkinter import ttk
from tkinter import font

if __name__ == '__main__':
    import farm.gui.components.pipelines as pipelines
    from farm.gui import IMAGE_FILES
else:
    from .components import pipelines


class App(tk.Tk):
    @staticmethod
    def logo():
        from pathlib import Path
        from PIL import Image, ImageTk
        
        logo_file = IMAGE_FILES['skao_logo']
        
        img = Image.open(logo_file)
        aspect = img.width / img.height
        
        img = ImageTk.PhotoImage(img.resize((100, int(100 / aspect))))

        return img
    
    def __init__(self, geometry: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry(geometry)
        
        # Creating a Font object of "TkDefaultFont"
        self.defaultFont = font.nametofont("TkDefaultFont")
        self.defaultFont.configure(family="Arial", size=13, weight=font.NORMAL)
        
        # Foundational frame to cover entire window
        self.window_frame = tk.Frame(padx=10, pady=10)
        self.window_frame.pack(fill='both', expand=True)

        # Child frames
        self.frame_logo = tk.Frame(self.window_frame)
        self.frame_pline_selection = tk.Frame(
            self.window_frame, padx=10, pady=10,
            highlightbackground="lightgrey", highlightthickness=1
        )
        self.hline = ttk.Separator(self.window_frame, orient='horizontal')
        self.frame_pline = pipelines.BlankPipelineOptions(
            self.window_frame, padx=0, pady=0
        )
        self.frame_buttons = tk.Frame(self.window_frame, padx=0, pady=0)
        
        # Attributes/variables
        self.pipeline_selection_label = ttk.Label(self.frame_pline_selection,
                                                  text="Select Pipeline:")
        
        self.pipeline_logo = ttk.Label(self.frame_logo)
        img_logo = self.logo()
        self.pipeline_logo.config(image=img_logo)
        self.pipeline_logo.image = img_logo
        
        self.pipeline_selection = ttk.Combobox(
            self.frame_pline_selection, state="readonly", 
            values=pipelines._AVAILABLE_PIPELINES
            )
        self.pipeline_selection.set("")
        self.pipeline_selection.bind('<<ComboboxSelected>>',
                                     self.change_pipeline)
        
        self._current_pipeline = None
        self._previous_pline = None
    
        # Buttons
        self.button_save = tk.Button(self.frame_buttons, text="Save")
        self.button_run = tk.Button(self.frame_buttons, text="Run")
        self.button_exit = tk.Button(self.frame_buttons, text="Exit",
                                     command=self.destroy)
    
        self.initiate()
        
    def initiate(self):
        """All gridding commands here to initiate"""
        self.frame_logo.pack(fill='x')
        self.frame_pline_selection.pack(fill='x')
        self.hline.pack(fill='x')
        self.frame_pline.pack(fill='both', expand=True)
        self.frame_buttons.pack(fill='x')

        self.pipeline_logo.pack(side=tk.RIGHT)

        self.pipeline_selection.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.pipeline_selection_label.pack(side=tk.LEFT, fill='none')

        self.frame_buttons.rowconfigure(0, weight=1)
        self.frame_buttons.columnconfigure(0, weight=1)
        self.frame_buttons.columnconfigure(1, weight=1)
        self.frame_buttons.columnconfigure(2, weight=1)
        self.frame_buttons.columnconfigure(3, weight=1)
        
        self.button_run.grid(column=0, row=0, sticky='ns', padx=0, pady=0)
        self.button_save.grid(column=1, row=0, sticky='ns', padx=0, pady=0)
        self.button_exit.grid(column=3, row=0, sticky='ns', padx=0, pady=0)

    @property
    def current_pipeline(self):
        return self.pipeline_selection.get()

    def change_pipeline(self, event: tk.Event):
        """Change pipeline being displayed in frame_pline"""
        if self._previous_pline == self.current_pipeline:
            return None

        self.frame_pline.destroy()
        self.frame_buttons.forget()
        
        new_frame = pipelines._PIPELINE_FRAMES[self.current_pipeline]
        self.frame_pline = new_frame(self.window_frame, bd=1, relief='groove')
        self.frame_pline.pack(fill='both', expand=True)
        self.frame_buttons.pack(fill='x')
        self._previous_pline = self.current_pipeline
        
        self.button_save.configure(command=self.frame_pline.save)
        self.button_run.configure(command=self.frame_pline.run)


if __name__ == '__main__':
    window = App(geometry='600x768')
    window.title('FARM')
    window.mainloop()
