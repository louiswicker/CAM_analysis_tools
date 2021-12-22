#!/usr/bin/env python

import sys
import os
import glob
import string
from optparse import OptionParser
subfolders = ['./']
# append path to the folder
sys.path.extend(subfolders)

parser = OptionParser()
parser.add_option("-f","--file",dest="file",type="string", help = "fortran file to be compiled - if not supplied, all fortran files are!")
parser.add_option("-a","--all",dest="all",default=False,help = "Boolean flag to compile all files (default=False)", action="store_true")
parser.add_option("-c","--compiler",dest="compiler",type="string",default=None,help = "compiler to use with f2py3")

(options, args) = parser.parse_args()

if options.all:
    fortran_files = glob.glob("*.f")
    fortran_files = fortran_files + glob.glob("*.f77")
    fortran_files = fortran_files + glob.glob("*.f90")

if options.file != None:
    prefix = options.file.split(".")[0]
    fortran_files = glob.glob(prefix + ".f90")
    fortran_files = fortran_files + glob.glob(prefix+".f")

if options.compiler == None:
    compiler = "gfortran"
else:
    compiler = options.compiler

libopts = "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib"

# go through all the folders, make the python module and run the program
#   f2py_module_list = []
#   run_module_list = []
#
# Get all the fortran files
#
for item in fortran_files:
    prefix = item.split(".")[0]
    print("\n=====================================================\n")
    print("  Attempting to compile file: %s " % item)
    print("\n=====================================================")
#   ret = os.system("f2py -c --fcompiler='gnu95' %s -m %s %s -DF2PY_REPORT_ON_ARRAY_COPY=1" % (libopts,prefix,item))
    ret = os.system("f2py -c --fcompiler='gnu95' %s -m %s %s" % (libopts,prefix,item))
    if ret == 0:
        print("\n=====================================================\n")
        print("   Successfully compiled file: %s " % item)
        print("   Object file is: %s " % (prefix + ".so"))
        print("\n======================================================")


#("f2py --debug --fcompiler=%s --f90flags='%s' %s -c -m %s %s %s %s " % (fopts['gnu'][0],fopts['gnu'][1], \
#               preprocess, item.split(".")[0], item, objects, fopts['gnu'][2]))
