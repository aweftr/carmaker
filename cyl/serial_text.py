import serial
import serial.tools.list_ports
import time

ports = list(serial.tools.list_ports.comports())
for i in ports:
    print(i[0])

ser = serial.Serial(port=ports[1][0])
while True:
    pip = 'e'
    print('All right')
    ser.write(pip.encode())
    pip = 'd'
    ser.write(pip.encode())

