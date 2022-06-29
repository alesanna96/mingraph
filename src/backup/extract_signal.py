from datasketch import MinHash
import numpy as np 
import pickle
import sys
import os
import json
sys.path.append(os.getcwd())
from utils.progress_monitor import progressbar

class SignalExtractor:
    def __init__(self,processed_exe_path):
        with open(processed_exe_path,'r') as exe_funcs:
            self.functions=json.load(exe_funcs)
        
        self.harmonic_freqs=pickle.load(open("./data/notes/notes_freqs.p","rb"))
        self.minhashes=[]
        self.signal=np.array([])

    def generate_minhash(self):
        try:
            for function_name,instructions in progressbar(self.functions.items(),prefix="minhashing progress: "):
                minhash_generator=MinHash(num_perm=108)
                for instruction in instructions:
                    minhash_generator.update(instruction.encode('utf8'))
                self.minhashes.append(minhash_generator.digest())
        except Exception as e:
            self.minhashes=[]
    
    def generate_signal(self):
        S_rate=16000
        T=1/S_rate
        t=0.01
        N=S_rate*t
        t_seq=np.arange(N)*T
        if len(self.minhashes)==0:
            self.signal=np.array([])
            return

        for encoding in self.minhashes:
            self.signal=np.hstack((self.signal,np.sum([amplitude*np.sin(2*np.pi*self.harmonic_freqs[i]*t_seq) for i,amplitude in enumerate(encoding)],axis=0)))
        
        self.N_tot=N*len(self.minhashes)
        self.signal_duration=np.arange(self.N_tot)*T 



"""
sampleRate = 44100
frequency = 440
length = 1

t = np.linspace(0, length, sampleRate * length)  #  Produces a 5 second Audio-File
y = np.sin(frequency* 2 * np.pi * t) + np.sin(7902* 2 * np.pi * t) + np.sin(698* 2 * np.pi * t)+ np.sin(82* 2 * np.pi * t)#  Has frequency of 440Hz
#plt.plot(t[:int(sampleRate/1000)],y[:int(sampleRate/1000)])
plt.plot(t,y)
plt.show()

S_rate=16000
T=1/S_rate
t=0.01
N=S_rate*t
freq=16.35
omega=2*np.pi*freq
t_seq=np.arange(N)*T
y1=np.sin(omega*t_seq)

"""

def parse_instruction(self,instr):
	if 'LAB' in instr:	
		if ',' in instr:
			return [instr.split()[0][:-1]]+[instr.split()[1]]+[instr.split(',')[0].replace(instr.split()[0]+' ','')]+[instr.split(',')[1]]
		else:
			instr_without_lab=instr[instr.index(' ')+1:]
			return [instr[:instr.index(' ')]]+[instr_without_lab[:instr_without_lab.index(' ')]]+[instr_without_lab[instr_without_lab.index(' '):]]
	else:
		if ',' in instr:
			return [instr.split()[0]]+[instr.split(',')[0].replace(instr.split()[0]+' ','')]+[instr.split(',')[1].strip()]
		else:
			return [instr[:instr.index(' ')],instr[instr.index(' '):].strip()]