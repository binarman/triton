#!/usr/bin/env python3
import csv
import sys

if len(sys.argv) < 2:
    bench_results_file = "bench_results.csv"
else:
    bench_results_file = sys.argv[1]

config_info = []
with open("configs") as configs_files:
    reader = csv.reader(configs_files, delimiter=',', quotechar='"')
    for row in reader:
        config_info += [row[1] + "x" + row[2] + " LDS write " + row[3] + " mfma" + row[5]]

# mapping from (config id, padding_config, counter name) to coutner value
counters = {}
# list of (config id, padding_config) ids
configs = []
# dict of config id -> [best bank conflict config, best align stalls, best config with sum of conflicts and stalls]
best_configs = {}
with open(bench_results_file) as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    # table contents:
    # "Kernel_Name", "VGPR_Count", "SGPR_Count", "Counter_Name", "Counter_Value"
    # kernel name format is <config id>__<interval1>_<pad1>_<interval2>_<pad2>
    for row in reader:
        kernel_name = row[0]
        config_id, padding_config = kernel_name.split('__')
        config_id = int(config_id)

        pad_components = padding_config.split('_')
        padding_config = ",".join([tup[0] + ":+" + tup[1] for tup in zip(pad_components[::2], pad_components[1::2])])

        vgprs = int(row[1])
        sgprs = int(row[2])
        counter_name = row[3]
        counter_val = float(row[4])
        counters[(config_id, padding_config, counter_name)] = (counter_val, vgprs)
        if (config_id, padding_config) not in configs:
            configs += [(config_id, padding_config)]
    for config_id, padding_config in configs:
        if config_id not in best_configs:
            best_configs[config_id] = {"conflict": ["", 1e10], "stall": ["", 1e10], "total": ["", 1e10]}
        conflicts = counters[(config_id, padding_config, "SQ_LDS_BANK_CONFLICT")][0]
        stalls = counters[(config_id, padding_config, "SQ_LDS_UNALIGNED_STALL")][0]

        padding_config = padding_config + " vgprs:" + str(
            counters[(config_id, padding_config, "SQ_LDS_UNALIGNED_STALL")][1])

        total = conflicts + stalls
        if best_configs[config_id]["conflict"][1] == conflicts:
            best_configs[config_id]["conflict"][0] += [padding_config]
        if best_configs[config_id]["stall"][1] == stalls:
            best_configs[config_id]["stall"][0] += [padding_config]
        if best_configs[config_id]["total"][1] == total:
            best_configs[config_id]["total"][0] += [padding_config]

        if best_configs[config_id]["conflict"][1] > conflicts:
            best_configs[config_id]["conflict"] = [[padding_config], conflicts]
        if best_configs[config_id]["stall"][1] > stalls:
            best_configs[config_id]["stall"] = [[padding_config], stalls]
        if best_configs[config_id]["total"][1] > total:
            best_configs[config_id]["total"] = [[padding_config], total]
    for config in best_configs:
        # print(config_info[config], "bank conflicts:", best_configs[config][0], "stalls:", best_configs[config][1], "total cycles:", best_configs[config][2])
        print(config_info[config], "total cycles:", best_configs[config]["total"])
