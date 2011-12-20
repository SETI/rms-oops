#!/usr/bin/python
import lbl_handler
from Tkinter import *
import tkMessageBox
import tkFileDialog
import simple_dialog
import vicar
import pylab

lh = lbl_handler.LBLHandler()

root = Tk()

#first set up gui to just load file & get info from each category of meta data
        

class TimeDialog(simple_dialog.Dialog):
    
    def body(self, master):
        
        Label(master, text="Minimum:").grid(row=0)
        Label(master, text="Maximum:").grid(row=1)
        
        self.e1 = Entry(master)
        self.e2 = Entry(master)
        
        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        return self.e1 # initial focus
    
    def apply(self):
        first_time_formatter = lbl_handler.TimeFormatter(self.e1.get())
        second_time_formatter = lbl_handler.TimeFormatter(self.e2.get())
        self.result = first_time_formatter, second_time_formatter

class ScrolledList(Listbox):
    def __init__(self, master, **kw):
        self.frame = Frame(master)
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)
        self.hbar = Scrollbar(self.frame, orient=HORIZONTAL)
        self.block = Frame(self.frame, width=18, height=18)
        self.block.grid(row=1, column=1)
        self.vbar = Scrollbar(self.frame, orient=VERTICAL)
        kw.setdefault('activestyle', 'none')
        kw.setdefault('highlightthickness', 0)
        if 'pack' in kw.keys() and kw.pop('pack') == 1:
            self.frame.pack(fill=BOTH, expand=1)
        Listbox.__init__(self, self.frame, **kw)
        self.grid(row=0, column=0, sticky=N+S+E+W)
        self.hbar.configure(command=self.xview)
        self.vbar.configure(command=self.yview)
        self.config(
                    yscrollcommand=self.vbar.set,
                    xscrollcommand=self.hbar.set
                    )
        self.hbar.grid(row=1, column=0, sticky=W+E)
        self.vbar.grid(row=0, column=1, sticky=N+S)
        
        self.pack = lambda **kw: self.frame.pack(**kw)
        self.grid = lambda **kw: self.frame.grid(**kw)
        self.place = lambda **kw: self.frame.place(**kw)
        self.pack_config = lambda **kw: self.frame.pack_config(**kw)
        self.grid_config = lambda **kw: self.frame.grid_config(**kw)
        self.place_config = lambda **kw: self.frame.place_config(**kw)
        self.pack_configure = lambda **kw: self.frame.pack_config(**kw)
        self.grid_configure = lambda **kw: self.frame.grid_config(**kw)
        self.place_configure = lambda **kw: self.frame.place_config(**kw)
    
    def gets(self):
        return self.get(0, END)
    
    def sets(self, arg):
        self.delete(0, END)
        try:
            arg = arg.strip('\n').splitlines()
        except AttributeError:
            pass
        if hasattr(arg, '__getitem__'):
            for item in arg:
                self.insert(END, str(item))
        else:
            raise TypeError("Scrolledlist.sets() requires a string of iterable of strings")

def callback():
    print "called the callback"

def quitCallback():
    status.set( "%s", "Asking to exit");
    if tkMessageBox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

def open_file_callback():
    # get filename
    filename = tkFileDialog.askopenfilename()

    # open file on your own
    if filename:
        status.set('Reading file %s', filename)
        #lh.read('./test_data/cassini/index.lbl')
        lh.read(filename)
        status.clear()
        rec_keys = lh.sorted_records()
        main_frame.lbl_list.delete(0, END)
        for obj in rec_keys:
            main_frame.lbl_list.insert(END, obj.data['NAME'])
        main_frame.lbl_list.update_idletasks()

def search_callback():
    category_obj = lh.lbl_objects[main_frame.current_metadata_category]
    if category_obj.data['DATA_TYPE'] == 'TIME':
        d = TimeDialog(root)
        ndx_list = lh.objects_between_times(d.result[0], d.result[1],
                                           main_frame.current_metadata_category)
        main_frame.image_data_list.select_clear(0)
        for i in ndx_list:
            main_frame.image_data_list.select_set(i)
        main_frame.image_data_list.yview(ndx_list[0])
        
    

class StatusBar(Frame):
    
    def __init__(self, master):
        Frame.__init__(self, master)
        self.label = Label(self, bd=1, relief=SUNKEN, anchor=W)
        self.label.pack(fill=X)
    
    def set(self, format, *args):
        self.label.config(text=format % args)
        self.label.update_idletasks()
    
    def clear(self):
        self.label.config(text="")
        self.label.update_idletasks()

class MainFrame:
    def __init__(self, master):
        frame = Frame(master, width=200, height=250)
        frame.pack()

        self.lbl_record_label = Label(frame, text="RECORDS:")
        self.lbl_record_label.pack()
        self.lbl_record_label.grid(row=0)
        self.lbl_list = ScrolledList(frame, width=50, height=40, pack=1)
        self.lbl_list.grid(row=1)
        self.lbl_list.insert(END, 'No Data')
        self.lbl_list.insert(END, '...')
        self.lbl_list.bind("<Double-Button-1>", self.data_type_chosen_callback)
    
        l = Label(frame, text="DATA:")
        l.pack()
        l.grid(row=0, column=1)
        self.image_data_list = ScrolledList(frame, width=50, height=40, pack=1,
                                            selectmode=EXTENDED)
        self.image_data_list.grid(row=1, column=1)
        self.image_data_list.insert(END, 'No Data')
        self.image_data_list.insert(END, '...')
        self.image_data_list.bind("<Double-Button-1>", self.object_chosen_callback)

        self.current_metadata_category = ''

    def data_type_chosen_callback(self, event):
        sel = self.lbl_list.curselection()
        for iter in sel:
            title = self.lbl_list.get(iter)
            obj = lh.lbl_objects[title]
            results = lh.table_column_info(obj.data['NAME'])
            self.image_data_list.delete(0, END)
            for line in results:
                self.image_data_list.insert(END, line)
            self.image_data_list.update_idletasks()
            self.current_metadata_category = title

    def object_chosen_callback(self, event):
        sel = self.image_data_list.curselection()
        for iter in sel:
            if self.current_metadata_category == 'FILE_NAME':
                title = self.image_data_list.get(iter)
                vimg = lh.image_data_for_file_name(title)
                pylab.gray()
                pylab.imshow(vimg.data[0])
            else:
                vimg = lh.image_data_for_row(iter)
                if vimg != None:
                    pylab.gray()
                    pylab.imshow(vimg.data[0])


root.protocol("WM_DELETE_WINDOW", quitCallback)

#create a menu
menu = Menu(root)
root.config(menu=menu)

filemenu = Menu(menu)
menu.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="Open...", command=open_file_callback)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=quitCallback)

editmenu = Menu(menu)
menu.add_cascade(label="Edit", menu=editmenu)
editmenu.add_command(label="Search...", command=search_callback)

helpmenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpmenu)
helpmenu.add_command(label="About...", command=callback)

main_frame = MainFrame(root)

status = StatusBar(root)
status.pack(side=BOTTOM, fill=X)

root.mainloop()
