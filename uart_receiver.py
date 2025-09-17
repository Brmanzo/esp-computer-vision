# uart_receiver.py
import serial, time, struct, os, sys

MAGIC = b'UBMP'

def crc16_modbus(data, seed=0xFFFF):
    crc = seed
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xA001) if (crc & 1) else (crc >> 1)
    return crc & 0xFFFF

def read_exact(ser, n):
    out = bytearray()
    while len(out) < n:
        chunk = ser.read(n - len(out))
        if not chunk:
            raise serial.SerialException("timeout/EOF while reading")
        out += chunk
    return bytes(out)

def open_port(port='/dev/ttyACM0', baud=921600):
    ser = serial.Serial(port, baudrate=921600, timeout=1,
                        rtscts=False, dsrdtr=False, xonxoff=False)
    # No DTR/RTS twiddling
    time.sleep(0.05)
    ser.reset_input_buffer()
    return ser

def recv_loop(port='/dev/ttyACM0', baud=921600, outdir="bmp_frames"):
    outdir = os.path.abspath(outdir)
    bad_dir = os.path.join(outdir, "_bad")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)
    print(f"[recv] saving to {outdir}")

    while True:
        try:
            ser = open_port(port, baud)
            print(f"[recv] port open: {port} @ {baud}")
            sync = b""
            while True:
                b = ser.read(1)
                if not b:
                    continue  # idle

                sync = (sync + b)[-4:]
                if sync == MAGIC:
                    # header OK
                    try:
                        raw_len = read_exact(ser, 4)
                        L = struct.unpack("<I", raw_len)[0]
                        if L == 0 or L > 10_000_000:
                            print(f"[recv] absurd length {L}, resync.")
                            sync = b""
                            continue

                        payload = read_exact(ser, L)
                        crc_rx_b = read_exact(ser, 2)
                        crc_rx = struct.unpack("<H", crc_rx_b)[0]
                        crc = crc16_modbus(payload)

                        print(f"[recv] UBMP len={L}, crc_rx=0x{crc_rx:04X}, crc_calc=0x{crc:04X}")

                        ts = int(time.time()*1000)
                        if crc == crc_rx:
                            path = os.path.join(outdir, f"frame_{ts}.bmp")
                            try:
                                with open(path, "wb") as f:
                                    f.write(payload)
                                print(f"[recv] saved {path} ({len(payload)} bytes)")
                            except Exception as ex:
                                print(f"[recv] save failed: {ex}")
                                # stash payload to debug perms/paths
                                with open(os.path.join(bad_dir, f"save_err_{ts}.bin"), "wb") as f:
                                    f.write(payload)
                        else:
                            print("[recv] CRC mismatch; quarantining frame")
                            with open(os.path.join(bad_dir, f"bad_crc_{ts}.bin"), "wb") as f:
                                f.write(payload)

                        sync = b""
                    except serial.SerialException as ex:
                        print(f"[recv] stream error: {ex}, resync")
                        sync = b""
                        time.sleep(0.2)

        except serial.SerialException as ex:
            print(f"[recv] reopen: {ex}")
            time.sleep(0.8)

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    recv_loop(port=port)