#!/usr/bin/python
################################################################################
# lbl_handler.py
#
# Classes and methods to deal with LBL and related files.
#
# Barton S. Wells, SETI Institute, December 15, 2011
################################################################################
import vicar
import os

################################################################################
# TimeFormatter class
#
# Class to handle the time format of the LBL data format metadata.
################################################################################
class TimeFormatter:
    
    ############################################################################
    # Constructor
    ############################################################################
    def __init__(self, date_string):
        """The constructor for a TimeFormatter object. It takes the string that
            LBL metadata uses for times and converts to constituent year, day
            number, hour, minute, second, and thousandths. Attempts to parse
            badly formed strings as well.
            
            Input:
            date_string     string in the form YYYY-DDDTHH:MM:SS.ttt where YYYY
                            refers to four-digit year, DDD refers to day number
                            in year, HH refers to hour in 24-hour clock, MM
                            refers to minute within hour, SS refers to second
                            within minute, and ttt refers to thousandths within
                            second. -,T,:, & . are literals.
            
            Return:         implicit return of class object.
            """
        dash = date_string.find('-')
        self.year = int(date_string[0:dash])
        t = date_string.find('T')
        self.day_number = int(date_string[dash+1:t])
        time_string = date_string[t+1:]
        colon = time_string.find(':')
        self.hours = 0
        self.minutes = 0
        self.seconds = 0
        self.decimals = 0
        self.thousandths = 0
        if colon > -1:
            self.hours = int(time_string[0:colon])
            time_string = time_string[colon+1:]
            colon = time_string.find(':')
            if colon > -1:
                self.minutes = int(time_string[0:colon])
                time_string = time_string[colon+1:]
        period = time_string.find('.')
        if period > -1:
            self.seconds = int(time_string[0:period])
            decimal_string = time_string[period+1:]
            exp = 3 - len(decimal_string)
            self.thousandths = int(decimal_string) * (10 ** exp)

    def __ge__(self, arg):
        """Determines whether self is later than or equal to arg, in time.
            called from self >= arg.
            
            Input:
            arg         another TimeFormatter object. right side of operand.
            
            Return:     self >= arg
            """
        if self.year > arg.year:
            return True
        elif self.year == arg.year:
            if self.day_number > arg.day_number:
                return True
            elif self.day_number == arg.day_number:
                if self.hours > arg.hours:
                    return True
                elif self.hours == arg.hours:
                    if self.minutes > arg.minutes:
                        return True
                    elif self.minutes == arg.minutes:
                        if self.seconds > arg.seconds:
                            return True
                        elif self.seconds == arg.seconds:
                            if self.thousandths >= arg.thousandths:
                                return True
        return False

    def __le__(self, arg):
        """Determines whether self is earlier than or equal to arg, in time.
            called from self <= arg.
            
            Input:
            arg         another TimeFormatter object. right side of operand.
            
            Return:     self <= arg
            """
        if self.year < arg.year:
            return True
        elif self.year == arg.year:
            if self.day_number < arg.day_number:
                return True
            elif self.day_number == arg.day_number:
                if self.hours < arg.hours:
                    return True
                elif self.hours == arg.hours:
                    if self.minutes < arg.minutes:
                        return True
                    elif self.minutes == arg.minutes:
                        if self.seconds < arg.seconds:
                            return True
                        elif self.seconds == arg.seconds:
                            if self.thousandths <= arg.thousandths:
                                return True
        return False


################################################################################
# ImageIndexTable class
#
# Class with data from image index table (*.tab csv files)
################################################################################
class ImageIndexTable:
    
    ############################################################################
    # Constructor
    ############################################################################
    def __init__(self, path, lbl_objects):
        """The constructor for a ImageIndexTable object. reads a *.tab LBL CSV
            table file and parses it into columns based on inputed LBL objects
            (column identifiers).
            
            Input:
            path            absolute path of *.tab file.
            lbl_objects     list of lbl_objects (column identifiers) of type
                            LBLObject.
            
            Return:         implicit return of class object.
            """
        # open file into single string
        input = open(path)
        file_string = input.read()
        input.close()
        self.data = {}
        lbl_keys = []
        # set up columns.  would be same as LBL Objects except there maybe be
        # multiple columns per LBL Object, in which case, identify columns
        for obj in lbl_objects:
            key = obj.data['NAME']
            if obj.data.has_key('ITEMS'):
                # we are working with multiple items for the same column,
                # so we subindex
                items_per_column = int(obj.data['ITEMS'])
                for i in range(items_per_column):
                    ckey = "%s_%d" % (key, i)
                    lbl_keys.append(ckey)
                    self.data[ckey] = []
            else:
                lbl_keys.append(key)
                self.data[key] = []
        # we have a simple csv file where columns relate to lbl_keys
        lines = file_string.split('\n')
        for line in lines:
            columns = self.csv_to_columns(line)
            i = 0
            for column in columns:
                entry = column.strip('"').strip()
                self.data[lbl_keys[i]].append(entry)
                i += 1

    def csv_to_columns(self, line):
        """split a csv line into columns. handles where commas exist within
            quotes.
            
            Input:
            line            a line of a csv file as separated from other lines
                            via a \n character
            
            Return:         a list of data from each column
            """
        dumb_columns = line.split(',')
        columns = []
        in_quote = False
        real_column = ''
        for column in dumb_columns:
            c = column.count('"')
            real_column = real_column + column
            if in_quote and c > 0:
                # this is the end of this real column
                columns.append(real_column)
                real_column = ''
                in_quote = False
            elif not in_quote:
                if (c % 2) == 0:
                    columns.append(column)
                    real_column = ''
                else:
                    in_quote = True
        return columns


################################################################################
# LBLObject class
#
# Class with metadata headers for what resides in each column of an
# ImageIndexTable.
################################################################################
class LBLObject:

    ############################################################################
    # Constructor
    ############################################################################
    def __init__(self, record):
        """The constructor for a LBLObject object. parses a record from a *.lbl
            file. a record is defined as that which defines a single column of
            a *.tab file.
            
            Input:
            record          metadata describing single column of *.tab file, in
                            the format of a *.lbl file.
            
            Return:         implicit return of class object.
            """
        self.data = {'NAME': 'Default'}
        lines = record.split('\n')
        for line in lines:
            words = line.split('=')
            if len(words) > 1 and words[0] != '':
                self.data[words[0].strip()] = words[1].strip()


################################################################################
# LBLHandler class
#
# Class to handle data from *.lbl inde files and the associated *.tab files.
################################################################################
class LBLHandler:
    
    ############################################################################
    # Constructor
    ############################################################################
    def __init__(self):
        """The constructor for a LBLHandler object. declares a dictionary of
            lbl_objects so that it exists if called.
            
            Return:         implicit return of class object.
            """
        self.lbl_objects = {}

    def read(self, path):
        """reads *.lbl index file, then reads index image table file referenced
            within file.
            
            Input:
            path            absolute path of *.lbl index file.
            """
        self.lbl_dir_name = os.path.dirname(path)  #save the directory
        input = open(path)
        file_string = input.read()
        input.close()
        records = file_string.split('  OBJECT         = COLUMN')
        objs = []
        for iter in range(len(records)):
            if iter < 1:
                self.header = LBLObject(records[0])
            else:
                obj = LBLObject(records[iter])
                #self.lbl_objects[obj.data['NAME']] = obj
                objs.append(obj)
        # now read the image index table file noted in header
        self.read_image_index_table(self.header.data['^IMAGE_INDEX_TABLE'],
                                    objs)
        # dictionaries do not keep order, so we need a look up table to match
        # row with the name of the row
        self.lbl_object_names = []
        for obj in objs:
            key = obj.data['NAME']
            self.lbl_objects[key] = obj
            self.lbl_object_names.append(key)

    def read_image_index_table(self, quoted_path, objs):
        """read the image index table file that is noted in this LBL file. the
            file name is likely in quotes, so be sure to strip the quotes from
            the file name first. also, the file name is relative to the
            directory of the LBL file, so find the directory from the LBL file
            name and prefix the relative file name."""
        relative_path = quoted_path.strip('"')
        path = self.lbl_dir_name + '/' + relative_path
        self.image_index_table = ImageIndexTable(path, objs)

    def sorted_records(self):
        """sort the names of the records, which can be thought of as the column
            titles for the image index table.
            
            Return:     list of titles, or keys to the dictionary records.
            """
        rec_keys = []
        for key in sorted(self.lbl_objects.iterkeys()):
            rec_keys.append(self.lbl_objects[key])
        return rec_keys

    def table_column_data(self, column_name):
        """get data for a specific column in the index image table.
            
            Input:
            column_name     title of column.
            
            Return:         list of column data.
            """
        obj = self.lbl_objects[column_name]
        col_data = []
        if obj.data.has_key('ITEMS'):
            # we are working with multiple items for the same column,
            # so we subindex
            items_per_column = int(obj.data['ITEMS'])
            row_data = []
            for i in range(items_per_column):
                ckey = "%s_%d" % (column_name, i)
                row_data.append(self.image_index_table.data[ckey])
            for i in range(len(row_data[0])):
                one_row = []
                for j in range(items_per_column):
                    one_row.append(row_data[j][i])
                col_data.append(one_row)
        else:
             col_data = list(self.image_index_table.data[column_name])
        return col_data


    def table_column_info(self, column_name):
        """get the information based on the data type for this column. if int or
            float, get min/max. if string, get list."""
        columns = self.table_column_data(column_name)
        data_type = str(type(columns[0][0]))
        ret_data = []
        if data_type.find('int') > -1:
            i = 0
            for column in columns:
                # we have an int, so find min/max
                ret_data[i] = "minimum = %d", min(column)
                i += 1
                ret_data[i] = "maximum = %d", max(column)
                i += 1
        elif data_type.find('float') > -1:
            i = 0
            for column in columns:
                # we have a float, so find min/max
                ret_data[i] = "minimum = %f", min(column)
                i += 1
                ret_data[i] = "maximum = %f", max(column)
                i += 1
        else:
            # we will assume strings, so return the entire list
            for column in columns:
                ret_data.append(column)
        return ret_data

    def image_data_for_row(self, row):
        """get the Vicar image for the specified row number in the index
            image table. looks up file name for that row and returns pixel
            data for that image.
            
            Input:
            row         row number, not necessarily an int (but will be
                        converted to an int)
            
            Return:     Vicar image.
            """
        irow = int(row)
        col_data = self.image_index_table.data['FILE_NAME']
        if irow < len(col_data):
            image_relative_path = col_data[irow]
            path = self.lbl_dir_name + '/' + image_relative_path
            vimg = vicar.VicarImage.from_file(path)
            return vimg
        return None
            
    def image_data_for_file_name(self, relative_path):
        """get the Vicar image for the specified file name. looks up file name
            for that row and returns pixel data for that image.
            
            Input:
            relative_path   file name, with path relative to the *.lbl index
                            file.
            
            Return:         Vicar image.
            """
        path = self.lbl_dir_name + '/' + relative_path
        vimg = vicar.VicarImage.from_file(path)
        return vimg

    ############################################################################
    # Methods to get data between minimums and maximums.
    # Separate methods for time data, float or int data, and string data.
    ############################################################################
    def objects_between_times(self, time0, time1, key):
        """get times between time0 and time1 from the column in the image index
            table specified by the title key.
            
            Input:
            time0       minimum time value for returned list.
            time1       maximum time value for returned list.
            key         title of column from image index table.
            
            Return:     list of indices refering to objects with times within
                        the specified range.
            """
        ndx_objects = []
        col_data = self.image_index_table.data[key]
        for i in range(len(col_data)):
            time_string = col_data[i]
            tf = TimeFormatter(time_string)
            if (tf >= time0) and (tf <= time1):
                ndx_objects.append(i)
            i += 1
        return ndx_objects
