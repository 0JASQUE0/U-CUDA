#include "ocr_client_win.h"
#include <stdexcept>
#include <windows.h>

static HANDLE H(void* p){ return reinterpret_cast<HANDLE>(p); }

OcrClient::OcrClient(const std::string& python_exe, const std::string& script){
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;          // дочерний процесс наследует хэндлы пайпов
    sa.lpSecurityDescriptor = nullptr;

    HANDLE inRd=nullptr, inWr=nullptr, outRd=nullptr, outWr=nullptr;

    // пайп для stdin питона: родитель пишет в inWr, питон читает из inRd
    if(!CreatePipe(&inRd, &inWr, &sa, 0))
        throw std::runtime_error("CreatePipe(stdin) failed");
    // конец, который остаётся у родителя (inWr), НЕ должен наследоваться
    SetHandleInformation(inWr, HANDLE_FLAG_INHERIT, 0);

    // пайп для stdout питона: питон пишет в outWr, родитель читает из outRd
    if(!CreatePipe(&outRd, &outWr, &sa, 0))
        throw std::runtime_error("CreatePipe(stdout) failed");
    SetHandleInformation(outRd, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput  = inRd;
    si.hStdOutput = outWr;
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE); // ошибки питона в консоль
    PROCESS_INFORMATION pi{};

    // командная строка: python ocr_server.py  (CreateProcessA меняет буфер,
    // поэтому копируем в модифицируемый вектор)
    std::string cmd = "\"" + python_exe + "\" \"" + script + "\"";
    std::vector<char> cmdbuf(cmd.begin(), cmd.end());
    cmdbuf.push_back('\0');

    BOOL ok = CreateProcessA(
        nullptr, cmdbuf.data(), nullptr, nullptr,
        TRUE,            // наследовать хэндлы
        CREATE_NO_WINDOW,// без отдельного консольного окна
        nullptr, nullptr, &si, &pi);

    // концы пайпов, принадлежащие дочернему процессу, родителю больше не нужны
    CloseHandle(inRd);
    CloseHandle(outWr);

    if(!ok){
        CloseHandle(inWr); CloseHandle(outRd);
        throw std::runtime_error("CreateProcess failed (python not found?)");
    }
    CloseHandle(pi.hThread);
    hProcess_       = pi.hProcess;
    hChildStdinWr_  = inWr;
    hChildStdoutRd_ = outRd;

    // ждём READY (или FATAL от сервера)
    std::string ready = read_line();
    if(ready.rfind("FATAL",0)==0)
        throw std::runtime_error("OCR model load failed: "+ready);
    if(ready!="READY")
        throw std::runtime_error("OCR server did not report READY (got: "+ready+")");
}

OcrClient::~OcrClient(){
    if(hChildStdinWr_){
        try { write_line("QUIT"); } catch(...) {}
        CloseHandle(H(hChildStdinWr_));
    }
    if(hProcess_){
        WaitForSingleObject(H(hProcess_), 3000);
        CloseHandle(H(hProcess_));
    }
    if(hChildStdoutRd_) CloseHandle(H(hChildStdoutRd_));
}

void OcrClient::write_line(const std::string& s){
    std::string line = s; line.push_back('\n');
    const char* p = line.data();
    DWORD left = (DWORD)line.size(), written = 0;
    while(left){
        if(!WriteFile(H(hChildStdinWr_), p, left, &written, nullptr) || written==0)
            throw std::runtime_error("write to OCR failed");
        p += written; left -= written;
    }
}

std::string OcrClient::read_line(){
    for(;;){
        auto nl = rbuf_.find('\n');
        if(nl != std::string::npos){
            std::string ln = rbuf_.substr(0, nl);
            rbuf_.erase(0, nl+1);
            if(!ln.empty() && ln.back()=='\r') ln.pop_back(); // CRLF от Windows
            return ln;
        }
        char tmp[4096]; DWORD n=0;
        if(!ReadFile(H(hChildStdoutRd_), tmp, sizeof(tmp), &n, nullptr) || n==0){
            if(!rbuf_.empty()){ std::string ln=rbuf_; rbuf_.clear(); return ln; }
            throw std::runtime_error("OCR process closed pipe");
        }
        rbuf_.append(tmp, n);
    }
}

std::string OcrClient::recognize_base64(const std::string& b64){
    write_line(b64);
    std::string r = read_line();
    if(r.rfind("OK ",0)==0)  return r.substr(3);
    if(r.rfind("ERR ",0)==0) throw std::runtime_error("OCR error: "+r.substr(4));
    throw std::runtime_error("OCR bad response: "+r);
}

bool OcrClient::ping(){ write_line("PING"); return read_line()=="OK pong"; }

std::string b64encode(const std::vector<unsigned char>& data){
    static const char* T="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out; out.reserve((data.size()+2)/3*4); size_t i=0;
    for(; i+3<=data.size(); i+=3){
        unsigned v=(data[i]<<16)|(data[i+1]<<8)|data[i+2];
        out+=T[(v>>18)&63]; out+=T[(v>>12)&63]; out+=T[(v>>6)&63]; out+=T[v&63];
    }
    size_t rem=data.size()-i;
    if(rem==1){ unsigned v=data[i]<<16; out+=T[(v>>18)&63]; out+=T[(v>>12)&63]; out+="=="; }
    else if(rem==2){ unsigned v=(data[i]<<16)|(data[i+1]<<8); out+=T[(v>>18)&63]; out+=T[(v>>12)&63]; out+=T[(v>>6)&63]; out+="="; }
    return out;
}
