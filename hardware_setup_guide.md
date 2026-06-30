# Hardware Setup Guide — SPQR-IoMT Testbed

## Required Hardware ($296 total)

| Item | Qty | Cost | Supplier |
|---|---|---|---|
| Raspberry Pi 4B (4GB) | 2 | $70 | raspberrypi.com |
| Raspberry Pi Pico | 2 | $8 | raspberrypi.com |
| STM32F446RE Nucleo board | 1 | $25 | mouser.com |
| INA219 breakout (Adafruit #904) | 4 | $20 | adafruit.com |
| 8-bit USB oscilloscope (Hantek 6022BE) | 1 | $45 | amazon |
| MicroSD cards 32GB (×2) | 2 | $16 | amazon |
| USB-C power supplies (5V/3A) | 2 | $14 | amazon |
| Breadboard + jumper wires | 1 set | $12 | amazon |
| 0.1 Ω shunt resistors | 4 | $6 | digikey |
| Ethernet cables (0.5m) | 3 | $9 | amazon |
| **Total** | | **$225** | |

---

## Raspberry Pi 4B Setup

### 1. Install OS
```bash
# Download Raspberry Pi OS Lite (64-bit)
# Flash to MicroSD with Raspberry Pi Imager
# Enable SSH in Imager settings before flashing
```

### 2. Install liboqs (Post-Quantum Crypto Library)
```bash
sudo apt update && sudo apt install -y \
    cmake ninja-build libssl-dev python3-dev python3-pip git

# Build liboqs
git clone --depth 1 https://github.com/open-quantum-safe/liboqs
cd liboqs && mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_SHARED_LIBS=ON ..
ninja && sudo ninja install
sudo ldconfig

# Install Python bindings
pip3 install liboqs-python --break-system-packages
```

### 3. Install SPQR-IoMT dependencies
```bash
git clone https://github.com/[author]/SPQR-IoMT
cd SPQR-IoMT
pip3 install -r requirements.txt --break-system-packages
```

### 4. Wire INA219 for Energy Measurement
```
RPi 4B GPIO    →    INA219 Breakout
─────────────────────────────────────
3.3V  (pin 1)  →    VCC
GND   (pin 6)  →    GND
SDA   (pin 3)  →    SDA
SCL   (pin 5)  →    SCL

Series circuit (for current measurement):
  PSU+ → INA219 V+ → 0.1Ω shunt → INA219 V- → DUT power rail
```

### 5. Verify INA219
```bash
sudo apt install -y python3-smbus i2c-tools
sudo i2cdetect -y 1
# Should show device at address 0x40

python3 -c "
from benchmarks.energy_meter import EnergyMeter
m = EnergyMeter(backend='ina219')
with m.measure('test') as ctx:
    import time; time.sleep(0.1)
print(ctx.result)
"
```

### 6. Run Crypto Benchmark
```bash
cd SPQR-IoMT
python3 experiments/exp1_crypto_overhead.py --iterations 200
# Results: benchmarks/results/exp1_crypto_overhead.csv
```

---

## STM32F446RE Nucleo Setup

### 1. Required Tools
```bash
# Linux
sudo apt install -y gcc-arm-none-eabi openocd gdb-multiarch
pip3 install pyocd --break-system-packages
```

### 2. Clone pqm4 (ARM Cortex-M4 Kyber)
```bash
git clone https://github.com/mupq/pqm4
cd pqm4
git submodule update --init --recursive

# Build Kyber512 for STM32F446
make PLATFORM=nucleo-f446re IMPLEMENTATION_PATH=crypto_kem/kyber512/m4
```

### 3. Flash and Benchmark
```bash
# Connect STM32 via USB
# Flash binary
openocd -f board/st_nucleo_f4.cfg \
        -c "program bin/kyber512_test.bin verify reset exit"

# Capture output via serial (115200 baud)
screen /dev/ttyACM0 115200
# Output: cycle counts for keygen/encaps/decaps
```

### 4. Energy Measurement on STM32
```
STM32 Nucleo  →  INA219
──────────────────────────
CN6 pin 1 (5V)  →  INA219 V+
0.1Ω shunt resistor in series
INA219 V-  →  CN6 pin 4 (GND side of DUT)
RPi I2C → INA219 SDA/SCL
```

```python
# Trigger STM32 measurement from RPi
import serial, time
from benchmarks.energy_meter import EnergyMeter

ser   = serial.Serial('/dev/ttyUSB0', 115200)
meter = EnergyMeter(backend='ina219')

# Send start command to STM32
with meter.measure('kyber512_m4_keygen') as ctx:
    ser.write(b'KEYGEN\n')
    while b'DONE' not in ser.readline():
        pass

print(f"Energy: {ctx.result.energy_uj:.2f} µJ")
print(f"Time:   {ctx.result.elapsed_ms:.2f} ms")
```

---

## Power Trace Collection (Side-Channel)

### Equipment
- Hantek 6022BE USB oscilloscope (8-bit, 1 MSPS max)
- SMA probe near STM32 power supply decoupling capacitor

### Software
```bash
pip3 install pyhantek6022 --break-system-packages
```

```python
# Collect power traces during Kyber keygen
from PyHT6022.LibUsbScope import Oscilloscope
import numpy as np

scope = Oscilloscope()
scope.setup()
scope.set_sample_rate(1)     # 1 MSPS
scope.set_ch1_voltage_range(2)  # ±2V

traces = []
for i in range(1000):
    # Trigger STM32 keygen
    # ...
    ch1, ch2 = scope.read_data(data_size=512)
    traces.append(ch1)

np.save('real_results/power_traces_kyber512.npy', np.array(traces))
print(f"Collected {len(traces)} traces")
```

---

## Network Testbed Wiring

```
[RPi Sensor 1] ──────┐
                      ├── [Ethernet Switch] ── [RPi Gateway]
[RPi Sensor 2] ──────┘

Run secure channel demo:
  # On gateway RPi:
  python3 -m pqc_layer.secure_channel  # starts server
  
  # On sensor RPi:
  python3 -c "
  from pqc_layer.secure_channel import run_demo
  run_demo('Kyber768')
  "
```

---

## Verification Checklist

- [ ] RPi 4B boots and SSH works
- [ ] liboqs installed: `python3 -c "import oqs; print(oqs.get_enabled_KEMs())"`
- [ ] INA219 detected at 0x40: `sudo i2cdetect -y 1`
- [ ] STM32 connected and flashable: `openocd -f board/st_nucleo_f4.cfg -c "init; reset; exit"`
- [ ] Oscilloscope detected: `python3 -c "from PyHT6022.LibUsbScope import Oscilloscope; s=Oscilloscope(); s.setup(); print('OK')"`
- [ ] Exp1 runs: `python3 experiments/exp1_crypto_overhead.py --iterations 10`
- [ ] Results saved: `ls benchmarks/results/`
