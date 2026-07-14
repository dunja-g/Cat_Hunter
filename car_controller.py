import serial
import time

# ============================================
# Serial configuration
# ============================================
SERIAL_PORT = '/dev/ttyACM0'   # Use 'COM3' etc. on Windows
BAUD_RATE = 9600


# ============================================
# Motor Controller class
# ============================================
class MotorController:
    """Control Arduino DC motors via serial port"""

    def __init__(self, port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=1):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)

    def _send(self, cmd):
        """Send command and print Arduino response"""
        self.ser.write(f"{cmd}\n".encode())
        time.sleep(0.3)
        while self.ser.in_waiting:
            response = self.ser.readline().decode().strip()
            print(f"  Arduino: {response}")

    def _send_fast(self, cmd):
        """Send command without reading response (for real-time RC)"""
        self.ser.write(f"{cmd}\n".encode())
        time.sleep(0.02)

    def forward(self, motor_num):
        print(f">>> Motor {motor_num} forward")
        self._send(f"MOTOR_{motor_num}_ON")

    def reverse(self, motor_num):
        print(f">>> Motor {motor_num} reverse")
        self._send(f"MOTOR_{motor_num}_REVERSE")

    def stop(self, motor_num):
        print(f">>> Motor {motor_num} stop")
        self._send(f"MOTOR_{motor_num}_OFF")

    def set_speed(self, motor_num, speed):
        print(f">>> Motor {motor_num} speed -> {speed}")
        self._send(f"MOTOR_{motor_num}_SPEED_{speed}")

    def drive(self, motor_num, direction, speed=None):
        """Drive single motor: direction = 'fwd' | 'rev' | 'stop',
           speed 0-255, keeps previous speed if not specified"""
        if speed is not None:
            self._send_fast(f"MOTOR_{motor_num}_SPEED_{speed}")
        if direction == 'fwd':
            self._send_fast(f"MOTOR_{motor_num}_ON")
        elif direction == 'rev':
            self._send_fast(f"MOTOR_{motor_num}_REVERSE")
        elif direction == 'stop':
            self._send_fast(f"MOTOR_{motor_num}_OFF")

    # ---- Car-level high-level commands (Arduino handles left/right) ----

    def car_set_speed(self, speed):
        """Set global speed"""
        self._send_fast(f"SPEED_{speed}")

    def car_forward(self):
        self._send_fast("FORWARD")

    def car_backward(self):
        self._send_fast("BACKWARD")

    def car_left(self):
        self._send_fast("LEFT")

    def car_right(self):
        self._send_fast("RIGHT")

    def car_stop(self):
        self._send_fast("STOP")

    def car_curve_left(self):
        self._send_fast("CLEFT")

    def car_curve_right(self):
        self._send_fast("CRIGHT")

    def car_curve_back_left(self):
        self._send_fast("CBLEFT")

    def car_curve_back_right(self):
        self._send_fast("CBRIGHT")

    def read_encoder(self, motor_num):
        print(f">>> Read encoder motor {motor_num}")
        self._send(f"MOTOR_{motor_num}_READ_ENCODER")

    def reset_encoder(self, motor_num):
        print(f">>> Reset encoder motor {motor_num}")
        self._send(f"MOTOR_{motor_num}_RESET_ENCODER")

    def stop_all(self):
        self._send_fast("MOTOR_1_STOP_ALL")

    # ---- Ultrasonic obstacle avoidance ----

    def car_set_threshold(self, cm):
        """Set obstacle distance threshold in cm (triggers emergency stop)"""
        self._send_fast(f"THRESHOLD_{cm}")

    def car_obstacle_on(self):
        """Enable ultrasonic obstacle avoidance"""
        self._send_fast("OBSTACLE_ON")

    def car_obstacle_off(self):
        """Disable ultrasonic obstacle avoidance"""
        self._send_fast("OBSTACLE_OFF")

    def car_read_distance(self):
        """Query ultrasonic distance, returns cm value, 999 on timeout"""
        self.ser.reset_input_buffer()
        self.ser.write(b"DISTANCE\n")
        # Wait for Arduino response "DIST:<value>"
        t0 = time.time()
        while time.time() - t0 < 0.5:
            if self.ser.in_waiting:
                line = self.ser.readline().decode(errors='replace').strip()
                if line.startswith("DIST:"):
                    try:
                        return float(line[5:])
                    except ValueError:
                        return -1
            time.sleep(0.01)
        return -2  # timeout

    def read_pending(self):
        """Non-blocking read of async messages from Arduino (OBSTACLE, etc.)
           Returns list of tuples: (type, value, extra?)
           e.g. ('OBSTACLE', 15.3, 'TURNING'), ('AVOID_DONE', None)"""
        messages = []
        while self.ser.in_waiting:
            try:
                line = self.ser.readline().decode(errors='replace').strip()
                if line == "AVOID_DONE":
                    messages.append(('AVOID_DONE', None))
                elif line.startswith("OBSTACLE:"):
                    rest = line[9:]  # "15.3" or "15.3:TURNING"
                    if ":TURNING" in rest:
                        parts = rest.split(":")
                        try:
                            dist = float(parts[0])
                        except ValueError:
                            dist = None
                        messages.append(('OBSTACLE', dist, 'TURNING'))
                    else:
                        try:
                            dist = float(rest)
                        except ValueError:
                            dist = None
                        messages.append(('OBSTACLE', dist))
                elif line.startswith("DIST:"):
                    try:
                        dist = float(line[5:])
                        messages.append(('DIST', dist))
                    except ValueError:
                        pass
                # Skip other debug messages
            except Exception:
                break
        return messages

    def close(self):
        self.ser.close()
        print("Serial closed")


# ============================================
# Interactive CLI
# ============================================
HELP_TEXT = """
Commands:
  f <n>              Motor n forward      (e.g. f 1)
  r <n>              Motor n reverse      (e.g. r 2)
  s <n>              Motor n stop         (e.g. s 1)
  speed <n> <0-255>  Motor n set speed     (e.g. speed 1 200)
  enc <n>            Read encoder n        (e.g. enc 1, motors 1-2 only)
  reset <n>          Reset encoder n       (e.g. reset 1, motors 1-2 only)
  stopall            Stop all motors
  help / ?           Show this help
  quit / exit / q    Quit
"""


def parse_speed(s):
    """Parse speed value 0-255"""
    try:
        v = int(s)
        if 0 <= v <= 255:
            return v
    except ValueError:
        pass
    return None


def parse_motor_num(s):
    """Parse motor number 1-4"""
    try:
        v = int(s)
        if 1 <= v <= 4:
            return v
    except ValueError:
        pass
    return None


def interactive_loop(mc: MotorController):
    print("=" * 50)
    print("  Motor Interactive Console")
    print("  Type 'help' for commands")
    print("=" * 50)

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        # ---- Help ----
        if cmd in ("help", "?"):
            print(HELP_TEXT)

        # ---- Quit ----
        elif cmd in ("quit", "exit", "q"):
            break

        # ---- Forward: f <n> ----
        elif cmd in ("f", "forward"):
            if len(parts) < 2:
                print("Usage: f <motor 1-4>")
                continue
            n = parse_motor_num(parts[1])
            if n is None:
                print("Invalid motor number (1-4)")
                continue
            mc.forward(n)

        # ---- Reverse: r <n> ----
        elif cmd in ("r", "reverse"):
            if len(parts) < 2:
                print("Usage: r <motor 1-4>")
                continue
            n = parse_motor_num(parts[1])
            if n is None:
                print("Invalid motor number (1-4)")
                continue
            mc.reverse(n)

        # ---- Stop: s <n> ----
        elif cmd in ("s", "stop"):
            if len(parts) < 2:
                print("Usage: s <motor 1-4>")
                continue
            n = parse_motor_num(parts[1])
            if n is None:
                print("Invalid motor number (1-4)")
                continue
            mc.stop(n)

        # ---- Set speed: speed <n> <0-255> ----
        elif cmd == "speed":
            if len(parts) < 3:
                print("Usage: speed <motor 1-4> <speed 0-255>")
                continue
            n = parse_motor_num(parts[1])
            if n is None:
                print("Invalid motor number (1-4)")
                continue
            v = parse_speed(parts[2])
            if v is None:
                print("Invalid speed value (0-255)")
                continue
            mc.set_speed(n, v)

        # ---- Encoder: enc <n> ----
        elif cmd in ("enc", "encoder"):
            if len(parts) < 2:
                print("Usage: enc <motor 1 or 2>")
                continue
            n = parse_motor_num(parts[1])
            if n is None or n > 2:
                print("Only motors 1 and 2 have encoders")
                continue
            mc.read_encoder(n)

        # ---- Reset encoder: reset <n> ----
        elif cmd == "reset":
            if len(parts) < 2:
                print("Usage: reset <motor 1 or 2>")
                continue
            n = parse_motor_num(parts[1])
            if n is None or n > 2:
                print("Only motors 1 and 2 have encoders")
                continue
            mc.reset_encoder(n)

        # ---- Stop all ----
        elif cmd in ("stopall", "allstop"):
            mc.stop_all()

        else:
            print(f"Unknown command: {cmd}. Type 'help' for help")


# ============================================
# Entry point
# ============================================
if __name__ == "__main__":
    try:
        mc = MotorController()
        interactive_loop(mc)
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        print("Check port and device connection")
    finally:
        try:
            mc.close()
        except Exception:
            pass
