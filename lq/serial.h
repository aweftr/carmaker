#pragma once
#include <Windows.h>

//get the com handle
HANDLE* openCom(const wchar_t* comName);

//init the com
bool initCom(HANDLE* handle);

//write to com
bool writeCom(HANDLE* handle);

//close com
bool closeCom(HANDLE* handle);