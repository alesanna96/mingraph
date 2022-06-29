import pefile
from capstone import *
from itertools import groupby
import re

example='/home/fra/Documents/thesis/data/samples/0a3e80dc7b5a8444a0579ee7af15004219d1ab7fa446d0bc1ab6a6c588f5b597'

class CodeXtractor:
    def __init__(self,exe_path):
        self.exe_path=exe_path
        self.pe=pefile.PE(self.exe_path)
        self.disassembler={'0x10b':Cs(CS_ARCH_X86, CS_MODE_32),\
                           '0x20b':Cs(CS_ARCH_X86,CS_MODE_64)}\
                            [hex(self.pe.OPTIONAL_HEADER.Magic)]
        self.pointer_size={'0x10b':'dword','0x20b':'qword'}[hex(self.pe.OPTIONAL_HEADER.Magic)]
        self.data=None 
        self.offset=None
        self.imports=dict([x for y in [[(hex(imp.address),imp.name.decode('ascii')) for imp in imports] 
        for imports in [entry.imports for entry in self.pe.DIRECTORY_ENTRY_IMPORT]] for x in y])
        
    def replace_address(self,instruction_arguments):
        for address,function_name in self.imports.items():
            if address in instruction_arguments:
                return instruction_arguments.replace(f'{self.pointer_size} ptr [{address}]', function_name)
        return instruction_arguments

    def extract_text_section(self):
        for section in self.pe.sections:
            if '0x'+hex(section.Characteristics)[2] in ['0x2','0x3','0x6','0x7'] and '0x'+hex(section.Characteristics)[-2] in ['0x2','0x3','0x6','0x7','0xa','0xb'] :
                self.data=self.pe.get_data(section.VirtualAddress,section.SizeOfRawData)
                self.offset=self.pe.OPTIONAL_HEADER.ImageBase+section.VirtualAddress
    
    def save_text_section(self,outpath):
        with open(outpath+self.exe_path.split('/')[-1] + "-" + 'text',"wb+") as data_out:
            data_out.write(self.data)
    
    def format_instruction(self,address,mnemonic,op_str,address_option=True):
        return {False:f'{hex(address)}: {mnemonic} {self.replace_address(op_str)}\n',\
                True:f'{mnemonic} {self.replace_address(op_str)}\n'}[address_option]

    def disassemble_text_section(self,remove_address=True):
        #self.code=[f'{hex(instruction.address)}: {instruction.mnemonic} {self.replace_address(instruction.op_str)}\n' for instruction in self.disassembler.disasm(self.data,self.offset)]
        self.code=[self.format_instruction(instruction.address,instruction.mnemonic,instruction.op_str,address_option=remove_address) for instruction in self.disassembler.disasm(self.data,self.offset)]
        self.code=[instr for instr, _ in groupby(self.code) if 'nop' not in instr]
    
    def save_disassembled_code(self,outpath):
        with open(outpath+self.exe_path.split('/')[-1] + "-" + 'code',"w") as data_out:
            data_out.writelines(self.code)



