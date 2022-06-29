* "Main" component is at ./src/process_samples and it is used to generate the json files containing the assembly instructions of each function of a sample. 

* Ghidra needs to be installed since this project uses ghidra scripts to extract the functions assembly.

* In the config file at ./config/config.json it is required to change the headless analyzer path (the used component of ghidra) to your install location.

* Load the project with the thesis folder as working directory.

* The extracted asm jsons, along with the malware samples and virus total jsons are put in the git ignore in order to ensure pushes are not too heavy. 

You can unzip these zip files to restore the folder content locally:
1. data/extracted_code => https://drive.google.com/file/d/1NLvY1V7sf-xKKPmjwbB6qw_A4tuM9Gay/view?usp=sharing
2. data/samples => https://ln.sync.com/dl/15e7527c0/p7idakza-vzm6xxqc-dxq6jnwa-hv2ar88t (these will contain also the virus total jsons, move them to data/vt_jsons)