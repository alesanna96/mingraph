import os
import json
import time

def timer(start,end):
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

with open('./config/config.json','r') as json_config:
        config=json.load(json_config)

"""
asm_ext=f'{config["headless_analyzer_path"]} \
        {config["temporary_project_folder"]} {config["temporary_project_name"]} \
        -import {config["samples_directory"]} \
        -analysisTimeoutPerFile {config["analysis_timeout"]} \
        -scriptPath {config["ghidra_scripts_path"]} \
        -preScript disable_decompiler.py \
        -postScript extract_funcs_asm.py {config["output_directory"]} \
        -readOnly \
        -deleteProject \
        -log {config["ghidra_script_log_path"]} \
        -scriptLog {config["python_script_log_path"]}'
"""

graph_ext=f'{config["ghidra_analyzer_script_settings"]["headless_analyzer_path"]} \
        {config["ghidra_analyzer_script_settings"]["temporary_project_folder"]} {config["ghidra_analyzer_script_settings"]["temporary_project_name"]} \
        -import {config["ghidra_analyzer_script_settings"]["samples_directory"]} \
        -analysisTimeoutPerFile {config["ghidra_analyzer_script_settings"]["analysis_timeout"]} \
        -scriptPath {config["ghidra_analyzer_script_settings"]["ghidra_scripts_path"]} \
        -preScript disable_decompiler.py \
        -postScript apply_static_lib_signatures.py {config["local_dbs"]["static_signatures_db"]} \
        -postScript extract_udf_calls_graph.py {config["ghidra_analyzer_script_settings"]["output_directory"]} \
        -deleteProject \
        -log {config["ghidra_analyzer_script_settings"]["ghidra_script_log_path"]} \
        -scriptLog {config["ghidra_analyzer_script_settings"]["python_script_log_path"]}'
start=time.time()
os.system(graph_ext)
end=time.time()
with open("/home/fra/Documents/thesis/utils/notes.txt","a") as notes:
        notes.write(f"upatre {timer(start,end)}\n")