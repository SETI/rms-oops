#!/usr/bin/python
from Tkinter import *
import tkMessageBox
import tkFileDialog
import simple_dialog
import vicar
import os
import pylab
import numpy as np
import oops
import pdstable
import julian
import math

CASSINI_label = 'CASSINI'
SATURN_label = 'SATURN'
J2000_label = 'J2000'
FILE_NAME_label = 'FILE_NAME'

root = Tk()

#first set up gui to just load file & get info from each category of meta data
oops.define_solar_system("2007-312T03:31:13.300", "2007-355T11:22:40.999")
#oops.define_cassini_saturn("2007-312T03:31:13.300", "2007-355T11:22:40.999")

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

class ImageWindow:
    def __init__(self):
        self.grid_on = False;
        self.gray_callback()
    
                       
    def autumn_callback(self):
        pylab.autumn()
                       
    def bone_callback(self):
        pylab.bone()
                       
    def cool_callback(self):
        pylab.cool()
                       
    def copper_callback(self):
        pylab.copper()
                       
    def flag_callback(self):
        pylab.flag()
                       
    def gray_callback(self):
        pylab.gray()
                       
    def hot_callback(self):
        pylab.hot()
                       
    def hsv_callback(self):
        pylab.hsv()
                       
    def jet_callback(self):
        pylab.jet()
                       
    def pink_callback(self):
        pylab.pink()
                       
    def prism_callback(self):
        pylab.prism()
                       
    def spring_callback(self):
        pylab.spring()
                       
    def summer_callback(self):
        pylab.summer()
                       
    def winter_callback(self):
        pylab.winter()
                       
    def spectral_callback(self):
        pylab.spectral()
                       
    def grid_callback(self):
        if self.grid_on:
            #redraw
            self.grid_on = False
        else:
            y = np.array([256., 512., 768.])
            pylab.hlines(y, 0., 1024., colors='r')
            pylab.vlines(y, 0., 1024., colors='r')
            self.grid_on = True

class ColumnInfoWindow:
    def __init__(self, master):
        self.parent = master

    def show(self, ptable, ndx):
        """show info window with column data.
            
            Input:
            ptable      PdsTable that has been loaded.
            ndx         row index from table.
            """
        self.top = Toplevel(master=self.parent)
        
        #show the file name that has been selected
        file_name_col = ptable.column_dict[FILE_NAME_label]
        file_name_label = "Filename: " + file_name_col[ndx]
        Label(self.top, text=file_name_label).pack()
        
        #compare file position to calculated position
        file_position_col = ptable.column_dict['SC_PLANET_POSITION_VECTOR']
        file_position_label = "File position: %f, %f, %f" % (file_position_col[ndx][0],
                                                             file_position_col[ndx][1],
                                                             file_position_col[ndx][2])
        Label(self.top, text=file_position_label).pack()

        time_col = ptable.column_dict['IMAGE_MID_TIME']
        tai = time_col[ndx]
        tdb = julian.tdb_from_tai(tai)

        event_at_cassini = oops.Event(tdb, (0,0,0), (0,0,0), CASSINI_label)
        saturn_wrt_cassini = oops.Path.connect(SATURN_label, CASSINI_label, J2000_label)
        (abs_event, rel_event) = saturn_wrt_cassini.photon_to_event(event_at_cassini)
        rel_pos_label = "Relative pos: %s" % rel_event.pos
        rel_vel_label = "Relative vel: %s" % rel_event.vel
        evt_at_cass_label = "Evt at Cassini arr: %s" % event_at_cassini.arr
        abs_event_label = "Abs evt dep: %s" % abs_event.dep
        Label(self.top, text=rel_pos_label).pack()
        Label(self.top, text=rel_vel_label).pack()
        Label(self.top, text=evt_at_cass_label).pack()
        Label(self.top, text=abs_event_label).pack()
        print rel_event.pos
        print rel_event.vel
        print event_at_cassini.arr
        print abs_event.dep

        sun_wrt_saturn = oops.Path.connect("SUN",SATURN_label)
        sun_dep_event = sun_wrt_saturn.photon_to_event(abs_event)
        print abs_event.arr
        print abs_event.phase_angle()
        abs_arr_label = "Abs evt arr: %s" % abs_event.arr
        phase_label = "Phase angle: %s" % abs_event.phase_angle()
        Label(self.top, text=abs_arr_label).pack()
        Label(self.top, text=phase_label).pack()
        
        #path = oops.SpicePath(SATURN_label, CASSINI_label, J2000_label)
        #better to use following
        path = oops.Path.connect(SATURN_label, CASSINI_label, J2000_label)
        event_at_cassini = path.event_at_time(tdb)
        (event_left_saturn, rel_event_left_saturn) = path.photon_to_event(event_at_cassini)
        print 'saturn pos:'
        print rel_event_left_saturn.pos
        print event_left_saturn.pos

        calc_pos_label = "Calculated position: %s" % rel_event_left_saturn.pos
        Label(self.top, text=calc_pos_label).pack()
        
        #do the ring plane
        #rings = oops.Frame.lookup("IAU_SATURN_DESPUN")
            #zaxis = oops.Event(tai, oops.Vector3([0,0,1]), oops.Vector3([0,0,0]),
        #CASSINI_label, rings)
        #zaxis_label = "Ring zaxis: %s" % zaxis
        #Label(self.top, text=zaxis_label).pack()
        
        b1 = Button(self.top, text="  OK  ", command=self.ok)
        b1.pack()
        b1.focus_set()

    def ok(self):
        self.top.destroy()

class MainFrame:
    def __init__(self, master):
        self.parent = master
        self.file_ndx = 0
        self.snapshot = None
        self.image_data = None
        
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
        self.lbl_list.bind("<d>", self.default_process_callback)
    
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
        
        image_window = ImageWindow()
        self.create_menu(master, image_window)
        master.protocol("WM_DELETE_WINDOW", self.quitCallback)

    
    def create_menu(self, master, image_window):
        menu = Menu(master)
        master.config(menu=menu)
                       
        filemenu = Menu(menu)
        menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open...", command=self.open_file_callback)
        filemenu.add_separator()
        filemenu.add_command(label="Info...", command=self.infoCallback)
        filemenu.add_separator()
        filemenu.add_command(label="Back-plane...", command=self.backplaneCallback)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quitCallback)
                       
        editmenu = Menu(menu)
        menu.add_cascade(label="Edit", menu=editmenu)
        editmenu.add_command(label="Search...", command=self.search_callback)
        editmenu.add_separator()
        
        cmapmenu = Menu(editmenu)
        editmenu.add_cascade(label="Colormap", menu=cmapmenu)
        cmapmenu.add_command(label="Autumn", command=image_window.autumn_callback)
        cmapmenu.add_command(label="Bone", command=image_window.bone_callback)
        cmapmenu.add_command(label="Cool", command=image_window.cool_callback)
        cmapmenu.add_command(label="Copper", command=image_window.copper_callback)
        cmapmenu.add_command(label="Flag", command=image_window.flag_callback)
        cmapmenu.add_command(label="Gray", command=image_window.gray_callback)
        cmapmenu.add_command(label="Hot", command=image_window.hot_callback)
        cmapmenu.add_command(label="HSV", command=image_window.hsv_callback)
        cmapmenu.add_command(label="Jet", command=image_window.jet_callback)
        cmapmenu.add_command(label="Pink", command=image_window.pink_callback)
        cmapmenu.add_command(label="Prism", command=image_window.prism_callback)
        cmapmenu.add_command(label="Spring", command=image_window.spring_callback)
        cmapmenu.add_command(label="Summer", command=image_window.summer_callback)
        cmapmenu.add_command(label="Winter", command=image_window.winter_callback)
        cmapmenu.add_command(label="Spectral", command=image_window.spectral_callback)
                       
        editmenu.add_separator()
        editmenu.add_command(label="Grid", command=image_window.grid_callback)
                    
        helpmenu = Menu(menu)
        menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="About...", command=callback)
        
        self.status = StatusBar(root)
        self.status.pack(side=BOTTOM, fill=X)

    def infoCallback(self):
        self.col_info_window = ColumnInfoWindow(self.parent)
        self.col_info_window.show(self.ptable, self.file_ndx)

    def quitCallback(self):
        self.status.set( "%s", "Asking to exit");
        if tkMessageBox.askokcancel("Quit", "Do you really wish to quit?"):
            root.destroy()

    def shape_for_instrument_mode(self, instrument_mode_ids):
        instrument_mode_id = instrument_mode_ids[self.file_ndx]
        if instrument_mode_id == 'FULL':
            return (1024,1024)
        elif instrument_mode_id == 'SUM2':
            return (512,512)
        elif instrument_mode_id == 'SUM4':
            return (256,256)
        else:
            return (128,128)

    def assign_to_snapshot(self):
        t0_col = self.ptable.column_dict['START_TIME']
        t1_col = self.ptable.column_dict['STOP_TIME']
        t0 = t0_col[self.file_ndx]
        t1 = t1_col[self.file_ndx]
        saturn_wrt_cassini_path = oops.Path.connect(SATURN_label, CASSINI_label, J2000_label)
        #uv_shape = self.shape_for_instrument_mode(self.ptable.column_dict['INSTRUMENT_MODE_ID'])
        fov = oops.FlatFOV(oops.Pair((math.pi/180/3600.,math.pi/180/3600.)),
                           (1024,1024))
        calibration = oops.Scaling("DN", 1.)
        
        #self.snapshot = oops.Snapshot(self.image_data, None, ["v","u"], (t0, t1),
        #                        fov, saturn_wrt_cassini_path, J2000_label,
        #                        calibration)
        self.snapshot = oops.Snapshot(self.image_data, None, ["v","u"], (t0, t1),
                                      fov, saturn_wrt_cassini_path, "SATURN_DESPUN",
                                      calibration)
        print "returning snapshot with t0 = %s" % self.snapshot.t0

    def in_ranges(self, value, ranges):
        for i in range(len(ranges)):
            if value >= ranges[i][0] and value <= ranges[i][1]:
                return True
        return False

    def create_sample_backplane_data(self):
        bp_data = np.empty((1024, 1024))
        offset = 28284.
        cutoff = 60298. * 60298.
        scale_pixel = 320.
        ring_ranges = [[66900.*66900., 74510.*74510.],
                       [74658.*74658., 92000.*92000.],
                       [92050.*92050., 117580.*117580.],
                       [122170.*122170., 136775.*136775.],
                       [170000.*170000., 175000.*175000.],
                       [181000.*181000., 483000.*483000.]]
        for i in range(1024):
            for j in range(1024):
                x = i * scale_pixel + offset
                y = j * scale_pixel + offset
                dist = x*x + y*y
                if self.in_ranges(dist, ring_ranges):
                    bp_data[i][j] = dist
                else:
                    bp_data[i][j] = 0.
        bp_data = np.sqrt(bp_data)
        return bp_data

    def backplaneCallback(self):
        """will create backplane then show it with pylab. for the moment, just
            show random data in pylab."""
        print "snapshot t0 = %s" % self.snapshot.t0
        if self.snapshot != None:
            bo_data = self.snapshot.radius_back_plane()
        else:
            bp_data = self.create_sample_backplane_data()
        d_max = bp_data.max()
        bp_data = bp_data / d_max
        pylab.imshow(bp_data)


    def open_file_callback(self):
        # get filename
        path = tkFileDialog.askopenfilename()
                       
        # open file on your own
        if path:
            self.status.set('Reading file %s', path)
            self.dir_path = os.path.dirname(path)  #save the directory
            self.ptable = pdstable.PdsTable(path, ['IMAGE_MID_TIME'])
            self.lbl_list.delete(0, END)
            for key in sorted(self.ptable.column_dict.iterkeys()):
                self.lbl_list.insert(END, key)
            self.lbl_list.update_idletasks()

    def default_process_callback(self, event):
        print 'in default process callback'
        path = '/Users/bwells/lsrc/pds-tools/test_data/cassini/ISS/index.lbl'
        self.dir_path = os.path.dirname(path)  #save the directory
        self.ptable = pdstable.PdsTable(path, ['IMAGE_MID_TIME'])
        self.lbl_list.delete(0, END)
        for key in sorted(self.ptable.column_dict.iterkeys()):
            self.lbl_list.insert(END, key)
        self.lbl_list.update_idletasks()
            
        iter = 0
        items = self.lbl_list.get(0, END)
        for title in items:
            if title == FILE_NAME_label:
                break
            iter += 1
        self.current_metadata_index = iter
        title = self.lbl_list.get(iter)
        self.image_data_list.delete(0, END)
        for line in self.ptable.column_dict[title]:
            self.image_data_list.insert(END, line)
        self.image_data_list.update_idletasks()
        self.current_metadata_category = title
        self.file_ndx = 3940
        self.open_image()

    def search_callback(self):
        column_info = self.ptable.info.column_info_list[self.current_metadata_index]
        category_obj = self.ptable.column_dict[self.current_metadata_category]
        if column_info.data_type == 'TIME':
            d = TimeDialog(root)
            ndx_list = self.ptable.columns_between(d.result[0], d.result[1],
                                                   self.current_metadata_index)
            self.image_data_list.select_clear(0)
            for i in ndx_list:
                self.image_data_list.select_set(i)
            self.image_data_list.yview(ndx_list[0])

    def data_type_chosen_callback(self, event):
        sel = self.lbl_list.curselection()
        for iter in sel:
            print 'iter = %d' % int(iter)
            self.current_metadata_index = iter
            title = self.lbl_list.get(iter)
            self.image_data_list.delete(0, END)
            for line in self.ptable.column_dict[title]:
                self.image_data_list.insert(END, line)
            self.image_data_list.update_idletasks()
            self.current_metadata_category = title

    def open_image(self):
        title = ""
        if self.current_metadata_category == FILE_NAME_label:
            title = self.image_data_list.get(self.file_ndx)
        else:
            col = self.ptable.column_dict[FILE_NAME_label]
            title = col[self.file_ndx]
        path = self.dir_path + '/' + title
        try:
            vimg = vicar.VicarImage.from_file(path)
        except IOError as e:
            print 'Image file does not exist on this system'
            return
        print vimg.table
        self.image_data = vimg.data[0]
        pylab.gray()
        pylab.imshow(self.image_data)
        #print out data
        for title in self.ptable.column_dict:
            col = self.ptable.column_dict[title]
            print "%s: %s" % (title, col[self.file_ndx])
        self.snapshot = oops.instrument.cassini.iss.from_file(path)

    def object_chosen_callback(self, event):
        sel = self.image_data_list.curselection()
        for iter in sel:
            print 'image iter = %d' % int(iter)
            self.file_ndx = int(iter)
            self.open_image()
            #self.print_data_for_row(self.file_ndx)
            self.assign_to_snapshot()

    def print_data_for_row(self, row):
        for key in sorted(self.ptable.column_dict.iterkeys()):
            col = self.ptable.column_dict[key]
            print '%s: %s' % (key, col[row])

#create a menu

main_frame = MainFrame(root)


root.mainloop()
