import serial
import serial.tools.list_ports
import time

print('hello')
ports = list(serial.tools.list_ports.comports())
print(ports)
pip = "hello";
ser = serial.Serial(port= ports[1][0])
print("get")
ser.write(pip.encode())
print('what')
