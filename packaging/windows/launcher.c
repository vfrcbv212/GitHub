#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <wchar.h>

/* Tiny native launcher. Python and all application dependencies ship beside it. */
int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR args, int show) {
    wchar_t root[32768], python[32768], command[32768];
    DWORD n = GetModuleFileNameW(NULL, root, 32768);
    if (!n || n >= 32768) return 2;
    wchar_t *separator = wcsrchr(root, L'\\');
    if (!separator) return 2;
    *separator = 0;
    if (swprintf(python, 32768, L"%ls\\runtime\\pythonw.exe", root) < 0 ||
        swprintf(command, 32768, L"\"%ls\" -I -B -X utf8 \"%ls\\app\\portable_bootstrap.py\" %ls", python, root, args) < 0) return 2;
    STARTUPINFOW startup = {0};
    PROCESS_INFORMATION process = {0};
    startup.cb = sizeof(startup);
    if (!CreateProcessW(python, command, NULL, NULL, FALSE, 0, NULL, root, &startup, &process)) {
        MessageBoxW(NULL, L"Не удалось запустить встроенный Python. Распакуйте весь ZIP и запускайте VolleyVision.exe из распакованной папки.", L"VolleyVision", MB_OK | MB_ICONERROR);
        return 2;
    }
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD status = 2;
    GetExitCodeProcess(process.hProcess, &status);
    CloseHandle(process.hProcess);
    return (int)status;
}
