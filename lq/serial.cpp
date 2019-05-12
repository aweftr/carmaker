#include "serial.h"

HANDLE* openCom(const wchar_t* comName) {
	HANDLE *SerialHandle = new HANDLE;

	//�򿪴���
	*SerialHandle = CreateFile(comName, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, 0);    //ͬ��ģʽ�򿪴���

	if (*SerialHandle == INVALID_HANDLE_VALUE)           //�򿪴���ʧ��
		return NULL;
	else
		return SerialHandle;
}

bool initCom(HANDLE* handle) {
	//���ô��ڲ���
//������ԭ���Ĳ�������
	DCB dcb;
	GetCommState(*handle, &dcb);   //���ڴ򿪷�ʽ

	//���ڲ������� 
	dcb.DCBlength = sizeof(DCB);
	dcb.BaudRate = 9600;
	dcb.ByteSize = 8;
	dcb.Parity = EVENPARITY;
	dcb.StopBits = ONESTOPBIT;
	dcb.fBinary = TRUE;                 //  ����������ģʽ
	dcb.fParity = TRUE;

	if (!SetCommState(*handle, &dcb))
		return false;

	SetupComm(*handle, 1024, 1024);    //���û�����
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