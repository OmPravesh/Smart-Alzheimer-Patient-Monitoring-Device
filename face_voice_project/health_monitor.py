# health_monitor.py - Fixed HR + clear text
import smbus2
import time
import numpy as np
from luma.core.interface.serial import i2c as i2c_serial
from luma.oled.device import sh1106
from luma.core.render import canvas

# ── OLED setup ──
serial = i2c_serial(port=1, address=0x3C)
oled   = sh1106(serial)

# ── MAX30102 ──
bus  = smbus2.SMBus(1)
ADDR = 0x57

def init_sensor():
    try:
        bus.write_byte_data(ADDR, 0x09, 0x40)  # Reset
        time.sleep(0.1)
        bus.write_byte_data(ADDR, 0x09, 0x03)  # SpO2 mode
        bus.write_byte_data(ADDR, 0x0A, 0x67)  # 100Hz, 16-bit
        bus.write_byte_data(ADDR, 0x0C, 0x7F)  # RED LED max
        bus.write_byte_data(ADDR, 0x0D, 0x7F)  # IR LED max
        bus.write_byte_data(ADDR, 0x08, 0x4F)  # FIFO config
        print("Sensor initialized!")
        return True
    except Exception as e:
        print("Init error:", e)
        return False

def read_raw_data(num_samples=300):
    red_vals = []
    ir_vals  = []
    for _ in range(num_samples):
        try:
            data = bus.read_i2c_block_data(ADDR, 0x07, 6)
            red  = (data[0] << 16 | data[1] << 8 | data[2]) & 0x3FFFF
            ir   = (data[3] << 16 | data[4] << 8 | data[5]) & 0x3FFFF
            red_vals.append(red)
            ir_vals.append(ir)
            time.sleep(0.01)
        except:
            pass
    return np.array(red_vals), np.array(ir_vals)

def calculate_hr(ir_data, fs=100):
    """FFT based heart rate"""
    if len(ir_data) < 100:
        return 0
    try:
        from scipy.signal import butter, filtfilt
        ac  = ir_data - np.mean(ir_data)
        nyq = fs / 2
        b, a = butter(2, [0.5/nyq, 4.0/nyq], btype='band')
        filtered  = filtfilt(b, a, ac)
        fft_vals  = np.abs(np.fft.rfft(filtered))
        freqs     = np.fft.rfftfreq(len(filtered), 1/fs)
        hr_range  = (freqs >= 0.5) & (freqs <= 4.0)
        if not np.any(hr_range):
            return 0
        peak_freq = freqs[hr_range][np.argmax(fft_vals[hr_range])]
        hr        = int(peak_freq * 60)
        if 40 <= hr <= 200:
            return hr
        return 0
    except Exception as e:
        print("HR error:", e)
        return 0

def calculate_spo2(red_data, ir_data):
    try:
        red_ac = np.std(red_data)
        red_dc = np.mean(red_data)
        ir_ac  = np.std(ir_data)
        ir_dc  = np.mean(ir_data)
        if ir_dc == 0 or red_dc == 0 or ir_ac == 0:
            return 0
        R    = (red_ac / red_dc) / (ir_ac / ir_dc)
        spo2 = int(104 - 17 * R)
        return max(80, min(100, spo2))
    except:
        return 0

# ── Rolling average ──
hr_history   = []
spo2_history = []

def smooth_reading(new_val, history, max_len=5):
    if new_val > 0:
        history.append(new_val)
        if len(history) > max_len:
            history.pop(0)
    return int(np.median(history)) if history else 0

# ── OLED display ──
def show_oled(hr, sp, status=""):
    with canvas(oled) as draw:

        if status == "waiting":
            draw.text((5,  10), "Place finger",  fill="white")
            draw.text((15, 30), "on sensor",     fill="white")
            draw.text((20, 50), "to start...",   fill="white")

        elif status == "measuring":
            draw.text((10, 20), "Measuring...",  fill="white")
            draw.text((5,  45), "Keep still!",   fill="white")

        else:
            # Title
            draw.text((20, 0),  "HEALTH MONITOR", fill="white")
            draw.line([(0, 12), (128, 12)],
                      fill="white", width=1)

            # Heart rate
            draw.text((0,  18), "Heart Rate:",   fill="white")
            hr_text = str(hr) + " BPM" if hr > 0 else "--- BPM"
            draw.text((0,  32), hr_text,         fill="white")

            # Divider
            draw.line([(0, 44), (128, 44)],
                      fill="white", width=1)

            # SpO2
            draw.text((0,  48), "SpO2:",         fill="white")
            sp_text = str(sp) + " %" if sp > 0 else "--- %"
            draw.text((50, 48), sp_text,         fill="white")

# ── MAIN ──
print("Starting health monitor...")
show_oled(0, 0, "waiting")
time.sleep(2)

# Install scipy if missing
try:
    from scipy.signal import butter, filtfilt
except ImportError:
    print("Installing scipy...")
    import subprocess
    subprocess.run(["pip", "install", "scipy"], check=True)

if not init_sensor():
    with canvas(oled) as draw:
        draw.text((5,  20), "Sensor Error!", fill="white")
        draw.text((5,  40), "Check wiring", fill="white")
    exit()

print("Place finger on sensor | Ctrl+C to stop")
FINGER_THRESHOLD = 50000

while True:
    try:
        # Quick finger check
        data     = bus.read_i2c_block_data(ADDR, 0x07, 6)
        ir_quick = (data[3] << 16 | data[4] << 8 | data[5]) & 0x3FFFF

        if ir_quick < FINGER_THRESHOLD:
            print("Waiting... IR:", ir_quick)
            show_oled(0, 0, "waiting")
            hr_history.clear()
            spo2_history.clear()
            time.sleep(0.3)
            continue

        # Finger detected
        print("Finger detected! Measuring...")
        show_oled(0, 0, "measuring")

        # Collect 300 samples
        red_data, ir_data = read_raw_data(300)

        # Calculate
        hr_raw   = calculate_hr(ir_data)
        spo2_raw = calculate_spo2(red_data, ir_data)

        # Smooth
        hr_smooth   = smooth_reading(hr_raw,   hr_history)
        spo2_smooth = smooth_reading(spo2_raw, spo2_history)

        print("HR:", hr_smooth, "BPM | SpO2:", spo2_smooth, "%")
        show_oled(hr_smooth, spo2_smooth)

        time.sleep(1)

    except KeyboardInterrupt:
        print("Stopped!")
        with canvas(oled) as draw:
            draw.text((20, 25), "Goodbye!", fill="white")
        break
    except Exception as e:
        print("Error:", e)
        time.sleep(1)