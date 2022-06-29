def disassemble_func2(func):
	listing=currentProgram.getListing()
	#func_end=getFunctionAfter(func).getEntryPoint().toString()
	instructions=[]
	labels=set()
	addresses=[]
	
	if func.isThunk():
		return instructions
		
	for index,instruction in enumerate(listing.getInstructions(func.getEntryPoint(),True)):
		try:
			addresses.append(instruction.getAddress().toString())
			instr_string=instruction.toString()
			if instr_string=="PUSH EBP" and index!=0: #instruction.getDefaultFallThrough().toString()==func_end:
				break
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
							if 'LAB' in replacement.toString():
								labels.add(replacement.toString())

			instructions.append(instr_string)
			
		except Exception as e:
			instructions.append(instruction.toString())	
	
	labels=list(labels)
	while len(labels)>0:
		address=addresses[addresses.index(labels[0][4:])]
		instructions.insert(addresses.index(address),labels[0]+':')
		addresses.insert(addresses.index(address),0)
		labels=labels[1:]
	return instructions


def export_disassembly(outpath='/home/fra/Documents/'):
	d=OrderedDict()
	func=getFirstFunction()
	while func is not None:
		d[func.getName()]=disassemble_func2(func)
		func=getFunctionAfter(func)
	
	with open(outpath+currentProgram.toString().split()[0]+'.json', 'w') as fp:
		json.dump(d, fp,indent=4)


export_disassembly()