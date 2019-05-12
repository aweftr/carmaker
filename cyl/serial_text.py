import serial
import serial.tools.list_ports
import time

print('hello')
ports = list(serial.tools.list_ports.comports())
print(ports)
pip = "a"
ser = serial.Serial(port= ports[1][0])

while True:
    ser.write(pip.encode())
    time.sleep(2)
