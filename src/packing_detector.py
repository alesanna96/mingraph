import pandas as pd
import sys,pefile,re,peutils,os,json

def detect_packing(pe,packer_signs_db):
	
	#check packer signatures presence
	if packer_signs_db.match_all(pe,ep_only = True):
		return 1
	
	# Entropy based check.. imported from peutils
	if peutils.is_probably_packed(pe):
		return 1

	# Entry point check	
	enaddr = pe.OPTIONAL_HEADER.AddressOfEntryPoint
	vbsecaddr = pe.sections[0].VirtualAddress
	ensecaddr = pe.sections[0].Misc_VirtualSize
	entaddr = vbsecaddr + ensecaddr
	if enaddr > entaddr:
		return 1
	
	return 0

def main():
	with open('./config/config.json','r') as json_config:
		config=json.load(json_config)
	
	packer_signs_db = peutils.SignatureDatabase(config["local_dbs"]["packer_signatures_db"])
	fname = config["packing_detector_settings"]["samples_directory"]
	if os.path.isdir(fname):
		filelist = os.listdir(fname)
		df={'name':[],'packed':[]}
		df_outpath=f'{config["packing_detector_settings"]["output_directory"]}{config["packing_detector_settings"]["output_name"]}'
		for i,name in enumerate(filelist):
			try:
				print(f"analysing {i+1}/{len(filelist)}")
				filename=name
				name = os.path.join(fname,name)
				pe = pefile.PE(name)
				packed = detect_packing(pe,packer_signs_db)
				pe.__data__.close()
				df['name'].append(filename)
				df['packed'].append(packed)
			except:
				pass
		df=pd.DataFrame(df)
		df.to_parquet(df_outpath)
	else:
		try:
			fname = os.path.realpath(fname)
			pe = pefile.PE(fname)
			packed = detect_packing(pe,packer_signs_db)
			print(packed)
			pe.__data__.close()
		except Exception:
			print("\nInvalid file\n")
			sys.exit(0)

if __name__ == '__main__':
		main()
