import serial
import serial.tools.list_ports
import time

ports = list(serial.tools.list_ports.comports())
ser = serial.Serial(port=ports[1][0])

angle = 180

j = angle // 4

while True:
    pip = input()
    ser.write(pip.encode())

