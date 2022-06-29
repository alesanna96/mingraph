#TODO write a description for this script
#@author 
#@category _NEW_
#@keybinding 
#@menupath 
#@toolbar 


#TODO Add User Code Here

from ghidra.util.task import ConsoleTaskMonitor
from  ghidra.app.decompiler import DecompInterface
from com.google.common.collect import Iterators
from ghidra.program.model.symbol import SourceType
import re
import json
import os
import time

def search_f(listing,fname):
	return [f for f in listing.getFunctions(True) if f.getName()==fname][0]

def select_pseudo_udf(f):
	return ' '.join([child.getName() for child in f.getCalledFunctions(ConsoleTaskMonitor()) if 'FUN' in child.getName()])

listing=currentProgram.getListing()
decompinterface = DecompInterface()
decompinterface.openProgram(currentProgram)

notfound_libfuncs=[f for f in listing.getFunctions(True) \
		if False in [True if calling.getName().find('FUN')>=0 \
		or calling.getName().find('case')>=0 \
		or calling.getName().find('entry')>=0 \
		else False \
		for calling in f.getCallingFunctions(ConsoleTaskMonitor())] and 'FUN' in f.getName()]

print "ALL LIB CALLED UDFS",len(notfound_libfuncs)

avoidable_notfound_libfuncs={f.getName() for f in notfound_libfuncs if len(f.getCalledFunctions(ConsoleTaskMonitor()))==0 or 'FUN' not in ' '.join([_.getName() for _ in f.getCalledFunctions(ConsoleTaskMonitor())])}
#unavoidable_notfound_libfuncs=[f for f in notfound_libfuncs if len(f.getCalledFunctions(ConsoleTaskMonitor()))>0]
fnames={f.getName() for f in notfound_libfuncs}.difference(avoidable_notfound_libfuncs)
all_calls=fnames

print "AVOIDABLE",len(avoidable_notfound_libfuncs)
print avoidable_notfound_libfuncs

start=time.time()
for i in range(200):
	fnames=set(' '.join(re.split("\W+",' '.join([select_pseudo_udf(search_f(listing,f)) for f in fnames]))).split()).difference(fnames)
	all_calls.update(fnames)
	current=time.time()-start
	if current>=2*60:
		break

#tokengrp = decompinterface.decompileFunction(function, 0, ConsoleTaskMonitor())
#float(Iterators.size(tokengrp.getHighFunction().getPcodeOps()))

print "DEPENDENCIES FOUND", len(all_calls)

renamings={f:list(search_f(listing,f).getCalledFunctions(ConsoleTaskMonitor()))[0].getName() \
	if Iterators.size(decompinterface.decompileFunction(search_f(listing,f), 0, ConsoleTaskMonitor()).getHighFunction().getPcodeOps())<10 \
	and len(list(search_f(listing,f).getCalledFunctions(ConsoleTaskMonitor())))==1 \
	else search_f(listing,f).getName() \
	for f in all_calls}

for i in range(20):
	renamings={f:renamings[substitute] if substitute in renamings.keys() else substitute for f,substitute in renamings.items()}


renamed=0
for name,renaming in renamings.items():
	if name!=renaming and "FUN" not in renaming:
		renamed+=1
		search_f(listing,name).setName(renaming, SourceType.USER_DEFINED)
		print name+"=>"+renaming
print "RENAMED",renamed

not_renamed={name for name,renaming in renamings.items() if name[:3]==renaming[:3]}
renamed={name for name,renaming in renamings.items() if name[:3]==renaming[:3] if name!=renaming and "FUN" not in renaming}

print set(notfound_libfuncs).difference(renamed).union(not_renamed)
print len(set(notfound_libfuncs).difference(renamed).union(not_renamed)) 

avoidable_notfound_libfuncs= set(notfound_libfuncs).difference(renamed).union(not_renamed)
print "TOTAL AVOIDED",len(avoidable_notfound_libfuncs)

print "generating graph..."
program_graph={}
program_graph["nodes_names"]=[]
program_graph["nodes"]=dict()
#all_neighbours=set()
for f in listing.getFunctions(True):
	if (f.getName()[:3]=="FUN" or f.getName()=="entry") and not f.isThunk() and f.getName() not in avoidable_notfound_libfuncs:
		program_graph["nodes_names"].append(f.getName())

		called_funcs={_.getName():_ if not _.isThunk() \
					    else list(_.getCalledFunctions(ConsoleTaskMonitor()))[0] \
					    	if len(list(_.getCalledFunctions(ConsoleTaskMonitor())))==1 \
						else _ \
			      for _ in f.getCalledFunctions(ConsoleTaskMonitor()) if _.getName() not in avoidable_notfound_libfuncs}
		for case in re.findall("case[^ ]*_[0-9]*",' '.join(called_funcs.keys())):
			#called_funcs.update({under_case_f.getName():under_case_f for under_case_f in called_funcs[case].getCalledFunctions(ConsoleTaskMonitor()) if under_case_f.getName() not in avoidable_notfound_libfuncs})
			called_funcs.update({_.getName():_ if not _.isThunk() \
					       		   else list(_.getCalledFunctions(ConsoleTaskMonitor()))[0] \
					    		   	if len(list(_.getCalledFunctions(ConsoleTaskMonitor())))==1 \
								else _ \
			      				   for _ in called_funcs[case].getCalledFunctions(ConsoleTaskMonitor()) \
							   if _.getName() not in avoidable_notfound_libfuncs})
			called_funcs.pop(case,None)
		
		#all_neighbours.update(set(re.findall("FUN_"+"[a-f0-9]"*8,' '.join(called_funcs.keys()))))
		program_graph["nodes"][f.getName()]=dict()
		program_graph["nodes"][f.getName()]["neighbourhood"]=re.findall("FUN_"+"[a-f0-9]"*8,' '.join(called_funcs.keys()))
		program_graph["nodes"][f.getName()]["features"]=list(set(called_funcs.keys()).difference(set(program_graph["nodes"][f.getName()]["neighbourhood"])))

#print "INTERSECTION", set(program_graph["nodes_names"]).intersection(avoidable_notfound_libfuncs)
#print "INTERSECTION", all_neighbours.intersection(avoidable_notfound_libfuncs)
#print len(all_neighbours),len(set(program_graph["nodes_names"]))

print "saving graph..."

outpath=getScriptArgs()[0]

with open(outpath+currentProgram.toString().split()[0]+'.json', 'w') as fp:
		json.dump(program_graph, fp,indent=4)

print "saved successfully"


