#include "serial.h"

HANDLE* openCom(const wchar_t* comName) {
	HANDLE *SerialHandle = new HANDLE;

	//打开串口
	*SerialHandle = CreateFile(comName, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, 0);    //同步模式打开串口

	if (*SerialHandle == INVALID_HANDLE_VALUE)           //打开串口失败
		return NULL;
	else
		return SerialHandle;
}

bool initCom(HANDLE* handle) {
	//设置串口参数
//读串口原来的参数设置
	DCB dcb;
	GetCommState(*handle, &dcb);   //串口打开方式

	//串口参数配置 
	dcb.DCBlength = sizeof(DCB);
	dcb.BaudRate = 9600;
	dcb.ByteSize = 8;
	dcb.Parity = EVENPARITY;
	dcb.StopBits = ONESTOPBIT;
	dcb.fBinary = TRUE;                 //  二进制数据模式
	dcb.fParity = TRUE;

	if (!SetCommState(*handle, &dcb))
		return false;

	SetupComm(*handle, 1024, 1024);    //设置缓冲区
	return true;
}

bool writeCom(HANDLE* handle) {
	DWORD WriteNum = 0;
	if (!WriteFile(*handle, "a", sizeof(char), &WriteNum, 0))
		return true;
	return false;
}


bool closeCom(HANDLE* handle) {
	if (*handle != INVALID_HANDLE_VALUE)
	{
		CloseHandle(*handle);
		*handle = INVALID_HANDLE_VALUE;
		return true;
	}
	else
		return false;
}