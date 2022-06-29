import pefile
from capstone import *

def disassemble(file_path):
    pe = pefile.PE(file_path)

    eop = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    code_section = pe.get_section_by_rva(eop)

    code_dump = code_section.get_data()
    
    code_addr = pe.OPTIONAL_HEADER.ImageBase + code_section.VirtualAddress

    md = {'0x10b':Cs(CS_ARCH_X86, CS_MODE_32),'0x20b':Cs(CS_ARCH_X86,CS_MODE_64)}[hex(pe.OPTIONAL_HEADER.Magic)]
    md.skipdata = True

    for i in md.disasm(code_dump, code_addr):
        #print("%i->0x%x:\t%s\t%s" %(j,i.address, i.mnemonic, i.op_str))
        print("%s\t%s" %(i.mnemonic, i.op_str))

disassemble('/home/fra/Documents/thesis/samples/exe_malware/0a0a36987ce13028b9d586ad002bfac3737b69f35bd5cadfd51fc493878f5744')