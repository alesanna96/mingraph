#Extracts the assembly from the user defined functions replacing all possible symbols and saves them in a json
#@author Francesco Meloni


from collections import OrderedDict
from copy import deepcopy
import json

def replace_symbol(instruction):
	instr_string=instruction.toString()
	try:
		if len(instr_string.split())!=1:
			if ',' in instr_string:
				ops=[instruction.getOpObjects(0)[0],instruction.getOpObjects(1)[0]]
			else:
				ops=[instruction.getOpObjects(0)[0]]
			for op in ops:
				if type(op)==ghidra.program.model.address.GenericAddress:
					replacement=getSymbolAt(op)
					if replacement is not None:
						instr_string=instr_string.replace('0x'+op.toString(),replacement.toString())
	except Exception as CouldNotReplaceSymbol:
		pass
	return instr_string


def add_internal_labels(instructions,internal_labels,addresses,labels_to_put_external):
	while len(internal_labels)>0:
		try:
			instructions[addresses.index(internal_labels[0])]='LAB_'+addresses[addresses.index(internal_labels[0])].toString()+': '+instructions[addresses.index(internal_labels[0])]
			internal_labels=internal_labels[1:]
		except ValueError as JumpToFunctionInsteadOfLabel:
			labels_to_put_external.add(internal_labels[0])
			internal_labels=internal_labels[1:]
	return instructions

def add_external_labels(exe_asm,labels):
	while len(labels)>0:
		print 'adding labels: '+str(len(labels))+' left'
		label=labels[0]
		for function,asm in exe_asm.items():
			try:
				exe_asm[function][label]='LAB_'+label.toString()+': '+asm[label] 		#asm[label]
				labels=labels[1:]
				break
			except KeyError as LabelNotinFunction:
				pass
		if len(labels)>0 and labels[0].toString()==label.toString():
			labels=labels[1:]
	

def remove_addresses(exe_asm):
	return OrderedDict([(function,asm.values()) for function,asm in exe_asm.items()])
					

def find_true_max_address(function):
	# some of these addresses can be out-of-function if the function contaisn jumps outside of it
	addresses=[address for address in function.getBody().getAddresses(True)] 
	# the entry of the next function is used in order to know which addresses are truly in the current function
	if getFunctionAfter(function) is not None:
		truly_contained=[address.subtract(getFunctionAfter(function).getBody().getMinAddress())<0 for address in addresses]
	else:
		max_address=function.getBody().getMaxAddress()
		if getInstructionAt(max_address) is None:
			max_address=getInstructionBefore(max_address).getAddress()
		return max_address
	if False in truly_contained:
		# next the first out-of-function address is found
		first_out_of_bounds=addresses[truly_contained.index(False)]
		# know we have a possible max address giving the we know its position
		max_address=addresses[addresses.index(first_out_of_bounds)-1]
	else:
		max_address=function.getBody().getMaxAddress()
	# sometimes this address can point to non-instruction data, so we can correct it by finding the true last instruction of the function
	if getInstructionAt(max_address) is None:
		max_address=getInstructionBefore(max_address).getAddress()
	# if max address doesn't count an eventual added part before the next function
	if getInstructionBefore(getFunctionAfter(function).getEntryPoint()).getAddress().subtract(max_address)>0:
		max_address=getInstructionBefore(getFunctionAfter(function).getEntryPoint()).getAddress()
	return max_address

def disassemble_function(listing,function,external_labels):
	instructions=[]
	addresses=[]
	internal_labels=set()
	max_address=find_true_max_address(function)
	
	for instruction in listing.getInstructions(function.getBody().getMinAddress(),True):
		# last instruction, process and break sweep
		if max_address==instruction.getAddress():
			instructions.append(replace_symbol(instruction))
			addresses.append(instruction.getAddress())
			if instruction.toString()[0]=='J':
				# save address of jump destination
				internal_labels.add(instruction.getOpObjects(0)[0])
			break
		# check if is jump istruction
		if instruction.toString()[0]=='J':
			# save address of jump destination
			internal_labels.add(instruction.getOpObjects(0)[0])
		instructions.append(replace_symbol(instruction))
		addresses.append(instruction.getAddress())
	instructions=add_internal_labels(instructions,list(internal_labels),addresses,external_labels)
	return OrderedDict(zip(addresses,instructions))


def export_disassembly(outpath):
	exe_asm=OrderedDict()
	labels=set()
	listing=currentProgram.getListing()
	for function in listing.getFunctions(True):
		if not function.isThunk() and '@' not in function.getName():
			print 'disassembling '+function.getName() +' of '+currentProgram.getName()
			exe_asm[function.getName()]=disassemble_function(listing,function,labels)
	add_external_labels(exe_asm,list(labels))
 
	with open(outpath+currentProgram.toString().split()[0]+'.json', 'w') as fp:
		json.dump(remove_addresses(exe_asm), fp,indent=4)

outpath=getScriptArgs()[0]
print outpath
export_disassembly(outpath)
