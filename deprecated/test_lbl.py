#!/usr/bin/python
from Tkinter import *
import tkMessageBox
import tkFileDialog
import simple_dialog
import vicar
import os
import pylab
import numpy as np
import numpy.ma as ma
import oops.tools as tools
from oops.event import Event
from oops.path.baseclass import Path
from oops.fov.flat import Flat
from oops.obs.snapshot import Snapshot
from oops.calib.scaling import Scaling
from oops.xarray.all import *
import oops.inst.cassini.iss
import oops.frame.all as frame_
import oops.path.all  as path_
import oops.surface.all as surface_
import oops.frame.registry as frame_registry
import oops.path.registry as path_registry
import pdstable
import julian
import math
import datetime
import cProfile

CASSINI_label = 'CASSINI'
SATURN_label = 'SATURN'
J2000_label = 'J2000'
FILE_NAME_label = 'FILE_NAME'
#DEFAULT_IMAGE_label = 'W1575634136_1.IMG'
DEFAULT_IMAGE_label = 'W1573721822_1.IMG'

root = Tk()

#first set up gui to just load file & get info from each category of meta data
tools.define_solar_system("2007-312T03:31:13.300", "2007-355T11:22:40.999")

    
def shadowProfileWrapper(obj, b):
    b.append(obj.snapshot.ring_shadow_back_plane(60298., 136775))

class AutoScrollbar(Scrollbar):
    # a scrollbar that hides itself if it's not needed.  only
    # works if you use the grid geometry manager.
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            # grid_remove is currently missing from Tkinter!
            self.tk.call("grid", "remove", self)
        else:
            self.grid()
        Scrollbar.set(self, lo, hi)
    def pack(self, **kw):
        raise TclError, "cannot use pack with this widget"
    def place(self, **kw):
        raise TclError, "cannot use place with this widget"

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

class ChooseFrameAndPath:
    def __init__(self, master):
        self.parent = master

    def show(self, frames, paths):
        self.top = Toplevel(master=self.parent)

        vscrollbar = Scrollbar(self.top, orient=VERTICAL)
        vscrollbar.pack(side=RIGHT, fill=Y)

        canvas = Canvas(self.top)
        canvas.pack()
        #canvas.grid(row=0, column=0, sticky=N+S+E+W)
        
        #
        # create canvas contents
        frame = Frame(canvas)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(1, weight=1)
        
        Label(frame, text="Frames").pack()
        for str in sorted(frames.iterkeys()):
            frames[str] = IntVar()
            frames[str].set(0)
            c = Checkbutton(frame, text=str, variable=frames[str], justify=LEFT)
            c.pack()
        Label(frame, text="Paths:").pack()
        for str in sorted(paths.iterkeys()):
            paths[str] = IntVar()
            paths[str].set(0)
            c = Checkbutton(frame, text=str, variable=paths[str], justify=LEFT)
            c.pack()
        b1 = Button(frame, text="  OK  ", command=self.ok)
        b1.pack()
        b1.focus_set()

        canvas.create_window(0, 0, anchor=NW, window=frame)
        canvas.config(yscrollcommand=vscrollbar.set)
        vscrollbar.config(command=canvas.yview)
        frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        """
        r = 1
        label = Label(self.parent, padx=7, pady=7, text="Frames:")
        label.grid(row=r, column=0, sticky='news')
        r += 1
        var = IntVar()
        for str in frames:
            c = Checkbutton(self.parent, text=str, variable=var, justify=LEFT)
            c.grid(row=r, column=0, sticky='news')
            r += 1

        label = Label(self.parent, padx=7, pady=7, text="Paths:")
        label.grid(row=r, column=0, sticky='news')
        r += 1
        for str in paths:
            c = Checkbutton(self.parent, text=str, variable=var, justify=LEFT)
            c.grid(row=r, column=0, sticky='news')
            r += 1

        b1 = Button(self.parent, text="  OK  ", command=self.ok)
        b1.grid(row=r, column=0, sticky='news')
        b1.focus_set()
            
        self.canvas.create_window(0, 0, anchor=NW, window=self.parent)
        self.parent.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
    """
        
    def ok(self):
        self.top.destroy()

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

        event_at_cassini = Event(tdb, (0,0,0), (0,0,0), CASSINI_label)
        saturn_wrt_cassini = Path.connect(SATURN_label, CASSINI_label, J2000_label)
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
        

        sun_wrt_saturn = Path.connect("SUN",SATURN_label)
        sun_dep_event = sun_wrt_saturn.photon_to_event(abs_event)
        print abs_event.arr
        print abs_event.phase_angle()
        abs_arr_label = "Abs evt arr: %s" % abs_event.arr
        phase_label = "Phase angle: %s" % abs_event.phase_angle()
        Label(self.top, text=abs_arr_label).pack()
        Label(self.top, text=phase_label).pack()
        
        #better to use following
        path = Path.connect(SATURN_label, CASSINI_label, J2000_label)
        event_at_cassini = path.event_at_time(tdb)
        (event_left_saturn, rel_event_left_saturn) = path.photon_to_event(event_at_cassini)
        print 'saturn pos:'
        print rel_event_left_saturn.pos
        print event_left_saturn.pos

        calc_pos_label = "Calculated position: %s" % rel_event_left_saturn.pos
        Label(self.top, text=calc_pos_label).pack()
        
        
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
            
        ignore = frame_.RingFrame("IAU_SATURN")
        self.frames = {}
        for key in frame_registry.REGISTRY:
            if 'SATURN' in key:
                if isinstance(key,str):
                    self.frames[key] = 0
                else:
                    self.frames[key[0]] = 0
        self.paths = {}
        for key in path_registry.REGISTRY:
            if 'SATURN' in key:
                if isinstance(key, str):
                    self.paths[key] = 0
                else:
                    self.paths[key[0]] = 0

    
    def create_menu(self, master, image_window):
        menu = Menu(master)
        master.config(menu=menu)
                       
        filemenu = Menu(menu)
        menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open Index...", command=self.open_file_callback)
        filemenu.add_command(label="Open Image...", command=self.open_image_file_callback)
        filemenu.add_command(label="Save as...", command=self.saveas_file_callback)
        filemenu.add_separator()
        filemenu.add_command(label="Info...", command=self.infoCallback)
        filemenu.add_separator()
        filemenu.add_command(label="Back-plane...", command=self.backplaneCallback)
        filemenu.add_command(label="Spheroid Back-plane...", command=self.spheroidBackplaneCallback)
        filemenu.add_command(label="Latitude Back-plane...", command=self.latitudeBackplaneCallback)
        filemenu.add_command(label="Shadow Back-plane...",
                             command=self.shadowBackplaneCallback)
        filemenu.add_command(label="Phase Angle Back-plane...",
                             command=self.phaseAngleBackplaneCallback)
        filemenu.add_command(label="Incidence Angle Back-plane...",
                             command=self.incidenceAngleBackplaneCallback)
        filemenu.add_command(label="Emission Angle Back-plane...",
                             command=self.emissionAngleBackplaneCallback)
        filemenu.add_separator()
        filemenu.add_command(label="Is Saturn in view?",
                             command=self.checkSaturnInViewCallback)
        filemenu.add_command(label="Center of Saturn in view?",
                             command=self.checkCenterSaturnInViewCallback)
        filemenu.add_command(label="Center of Objects in view?",
                             command=self.checkCenterOfObjectsInViewCallback)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quitCallback)
                       
        editmenu = Menu(menu)
        menu.add_cascade(label="Edit", menu=editmenu)
        editmenu.add_command(label="Search...", command=self.search_callback)
        editmenu.add_separator()
        editmenu.add_command(label="Choose Frames and Paths...",
                             command=self.chooseFrameAndPathsCallback)
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

    def chooseFrameAndPathsCallback(self):
        cfp = ChooseFrameAndPath(self.parent)
        cfp.show(self.frames, self.paths)

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
        saturn_wrt_cassini_path = Path.connect(SATURN_label, CASSINI_label, J2000_label)
        #uv_shape = self.shape_for_instrument_mode(self.ptable.column_dict['INSTRUMENT_MODE_ID'])
        fov = Flat(oops.Pair((math.pi/180/3600.,math.pi/180/3600.)),
                           (1024,1024))
        calibration = Scaling("DN", 1.)
        
        self.snapshot = Snapshot(self.image_data, None, ["v","u"], (t0, t1),
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

    def displayBackplane(self, bp_data):
        """display in pylab image the backplane data"""
        bp_data = np.nan_to_num(bp_data)
        d_max = bp_data.max()
        print "d_max:"
        print d_max
        d_min = bp_data.min()
        #d_min = 9999999999.
        #for y in range(1024):
            #for x in range(1024):
                #if bp_data[y][x] > 0. and d_min > bp_data[y][x]:
                    #d_min = bp_data[y][x]
        print "d_min:"
        print d_min
        bp_data = bp_data / d_max
        print "normalized bp_data:"
        print bp_data
        pylab.imshow(bp_data)
        self.img_data = bp_data

    def backplaneCallback(self):
        """will create backplane then show it with pylab. for the moment, just
            show random data in pylab."""
        print "snapshot t0 = %s" % self.snapshot.t0
        bp_data = None
        if self.snapshot != None:
            print "creating radius back plane..."
            bp_data = self.snapshot.radius_back_plane(60298., 136775.)
        else:
            bp_data = self.create_sample_backplane_data()
        self.displayBackplane(bp_data)
        pylab.imsave("/Users/bwells/backplane.png", bp_data)

    def spheroidBackplaneCallback(self):
        """will create backplane then show it with pylab. for the moment, just
        show random data in pylab."""
        bp_data = None
        if self.snapshot != None:
            print "creating polar back plane..."
            bp_data = self.snapshot.polar_back_plane("SATURN", "IAU_SATURN")
        else:
            bp_data = self.create_sample_backplane_data()
        self.displayBackplane(bp_data)
        pylab.imsave("/Users/bwells/polarbackplane.png", bp_data)

    def latitudeBackplaneCallback(self):
        """will create backplane then show it with pylab. for the moment, just
            show random data in pylab."""
        bp_data = None
        if self.snapshot != None:
            print "creating latitude back plane..."
            bp_data = self.snapshot.latitude_c_back_plane("SATURN",
                                                          "IAU_SATURN")
        else:
            bp_data = self.create_sample_backplane_data()
        self.displayBackplane(bp_data)
        pylab.imsave("/Users/bwells/latitudebackplane.png", bp_data)

    def shadowBackplaneCallback(self):
        """will create backplane then show it with pylab. for the moment, just
            show random data in pylab."""
        bp_data = None
        then = datetime.datetime.now()
        if self.snapshot != None:
            print "creating shadow back plane..."
            bp_data = self.snapshot.ring_shadow_back_plane(60298., 136775)
        else:
            bp_data = self.create_sample_backplane_data()
        self.displayBackplane(bp_data)
        now = datetime.datetime.now()
        total_time = now - then
        print "ring shadow back plane took:"
        print str(total_time)
        #pylab.imsave("/Users/bwells/latitudebackplane.png", bp_data)

    def phaseAngleBackplaneCallback(self):
        """will create backplane then show it with pylab. for the moment, just
            show random data in pylab."""
        bp_data = None
        then = datetime.datetime.now()
        if self.snapshot != None:
            print "creating phase angle back plane..."
            pa = self.snapshot.phase_angle_back_plane(60298., 136775)
            bp_data = pa.mvals
        else:
            bp_data = self.create_sample_backplane_data()
        self.displayBackplane(bp_data)
        now = datetime.datetime.now()
        total_time = now - then
        print "phase angle back plane took:"
        print str(total_time)
        #pylab.imsave("/Users/bwells/latitudebackplane.png", bp_data)

    def incidenceAngleBackplaneCallback(self):
        """will create backplane then show it with pylab. for the moment, just
            show random data in pylab."""
        bp_data = None
        then = datetime.datetime.now()
        if self.snapshot != None:
            print "creating incidence angle back plane..."
            pa = self.snapshot.incidence_angle_back_plane(60298., 136775)
            #now change the data to more accurately look like diffuse shading
            x = np.cos(pa.vals)
            x[x<0.] = 0.
            bp_data = ma.array(x, mask=pa.mask)
        else:
            bp_data = self.create_sample_backplane_data()
        self.displayBackplane(bp_data)
        now = datetime.datetime.now()
        total_time = now - then
        print "incidence angle back plane took:"
        print str(total_time)

    def emissionAngleBackplaneCallback(self):
        """will create backplane then show it with pylab. for the moment, just
            show random data in pylab."""
        bp_data = None
        then = datetime.datetime.now()
        if self.snapshot != None:
            print "creating emission angle back plane..."
            pa = self.snapshot.emission_angle_back_plane(60298., 136775)
            #now change the data to more accurately look like diffuse shading
            x = np.cos(pa.vals)
            x[x<0.] = 0.
            bp_data = ma.array(x, mask=pa.mask)
        else:
            bp_data = self.create_sample_backplane_data()
        self.displayBackplane(bp_data)
        now = datetime.datetime.now()
        total_time = now - then
        print "emission angle back plane took:"
        print str(total_time)

    def checkSaturnInViewCallback(self):
        ignore = frame_.RingFrame("IAU_SATURN")
        surface = surface_.Spheroid("SATURN", "IAU_SATURN", 60268., 54364.)
        in_view = self.snapshot.surface_in_view(surface)
        print "Center of Saturn in view:"
        print in_view

    def checkCenterOfObjectsInViewCallback(self):
        #cheating b/c don't know how to get sizes of bodies
        #cheat_a = 62.8
        #cheat_c = 39.7
        #objects_in_view = []
        #objects_not_in_view = []
        for fkey in self.frames:
            if self.frames[fkey].get():
                for pkey in self.paths:
                    if self.paths[pkey].get():
                        print "%s, %s" % (fkey, pkey)
                        #surface = surface_.Spheroid(pkey, fkey, cheat_a, cheat_c)
                        in_view = self.snapshot.surface_center_within_view_bounds(pkey)
                        print "%s in view:" % pkey
                        print in_view
        """fkey = "IAU_SATURN"
        for pkey in self.paths:
            print "%s, %s" % (fkey, pkey)
            surface = surface_.Spheroid(pkey, fkey, cheat_a, cheat_c)
            in_view = self.snapshot.surface_center_within_view_bounds(surface)
            if in_view:
                objects_in_view.append(pkey)
            else:
                objects_not_in_view.append(pkey)
        print "objects in view:\n------------"
        print objects_in_view
        print "objects NOT in view:\n------------"
        print objects_not_in_view"""

    def checkCenterSaturnInViewCallback(self):
        #ignore = frame_.RingFrame("IAU_SATURN")
        #surface = surface_.Spheroid("SATURN", "IAU_SATURN", 60268., 54364.)
        in_view = self.snapshot.surface_center_within_view_bounds("SATURN")
        print "Center of Saturn in view:"
        print in_view
        print "Any part of Saturn in view:"
        print self.snapshot.any_part_object_in_view("SATURN")

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

    def saveas_file_callback(self):
        # get filename
        path = tkFileDialog.asksaveasfilename()
    
        # save file of image data
        if path:
            pylab.imsave(path, pylab.gci().get_array())

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
        f_ndx = 0
        for line in self.ptable.column_dict[title]:
            self.image_data_list.insert(END, line)
            if DEFAULT_IMAGE_label in line:
                self.file_ndx = f_ndx
            f_ndx += 1
        self.image_data_list.update_idletasks()
        self.current_metadata_category = title
        #self.file_ndx = iter
        print "self.file_ndx = %d" % self.file_ndx
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

    def open_image_file_callback(self):
        path = tkFileDialog.askopenfilename()
        try:
            vimg = vicar.VicarImage.from_file(path)
        except IOError as e:
            print 'Image file does not exist on this system'
            return
        self.snapshot = oops.inst.cassini.iss.from_file(path)
        for item in vimg.table:
            print item
        #print vimg.table
        self.image_data = vimg.data[0]
        pylab.gray()
        pylab.imshow(self.image_data)
        pylab.imsave("/Users/bwells/saturnImage.png", self.image_data)
        ignore = frame_.RingFrame("IAU_SATURN")

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
        pylab.imsave("/Users/bwells/saturnImage.png", self.image_data)
        #print out data
        for title in sorted(self.ptable.column_dict.iterkeys()):
            col = self.ptable.column_dict[title]
            print "%s: %s" % (title, col[self.file_ndx])
        self.snapshot = oops.inst.cassini.iss.from_file(path)

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

#cProfile.run('root.mainloop()')
root.mainloop()
