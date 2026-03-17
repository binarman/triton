#!/opt/venv/bin/python3
import os
import sys

additional_path = "/opt/rocm/libexec/rocm_smi"
sys.path.append(additional_path)
try:
    from rsmiBindings import *
except ImportError:
    print(f"Couldn't import 'rsmiBindings'. Make sure it's installed in {additional_path}")
    sys.exit(1)

rocmsmi = initRsmiBindings(silent=False)
ret_init = rocmsmi.rsmi_init(0)


# Returns value in milli Joules
def GetEnergy(device):
    power = c_uint64()
    timestamp = c_uint64()
    counter_resolution = c_float()
    ret = rocmsmi.rsmi_dev_energy_count_get(device, byref(power), byref(counter_resolution), byref(timestamp))
    if ret != rsmi_status_t.RSMI_STATUS_SUCCESS:
        print("Couldn't get energy consumption")
        return None
    return power.value * counter_resolution.value


def getGPUFreq(device):
    freq = rsmi_frequencies_t()
    rocmsmi.rsmi_dev_gpu_clk_freq_get(device, rsmi_clk_names_dict['sclk'], byref(freq))
    lvl = freq.current
    freqMhz = freq.frequency[lvl] / 1000000
    return freqMhz


def getIdlePowerConsumption(repeats, device_idx):
    total_consumed = 0.0
    for i in range(repeats):
        start_energy = GetEnergy(device_idx)
        # here goes matmul(a, b)
        total_consumed += GetEnergy(device_idx) - start_energy
    return total_consumed


if __name__ == "__main__":
    import triton
    device_idx = triton.runtime.driver.active.get_active_torch_device().index

    startEnergy = GetEnergy(device_idx)
    dtime = 2.0
    import time
    time.sleep(dtime)
    endEnergy = GetEnergy(device_idx)

    print("consumed {} Joules over {} seconds".format((endEnergy - startEnergy) / 1e6, dtime))
    if dtime != 0.0:
        print("average power {} Watt".format((endEnergy - startEnergy) / 1e6 / dtime))
    else:
        print("average power N/A, time delta is zero")

    dtime = 1.0
    num_experiments = 1000
    measurements = []
    timing = []
    for i in range(num_experiments):
        start_energy = GetEnergy(device_idx)
        start_time = time.time()
        # time.sleep(dtime/(num_experiments-1))
        end_energy = GetEnergy(device_idx)
        end_time = time.time()
        measurements += [(endEnergy - startEnergy) / 1000000]
        timing += [end_time - start_time]
    print("min max power measured", min(measurements), max(measurements))
    print("min max time spend on power measurements", min(timing), max(timing))

    freq = rsmi_frequencies_t()
    rocmsmi.rsmi_dev_gpu_clk_freq_get(device_idx, rsmi_clk_names_dict['sclk'], byref(freq))
    lvl = freq.current
    freqMhz = freq.frequency[lvl] / 1000000
    print("freq lvl", lvl, "value:", freqMhz, "Mhz")
