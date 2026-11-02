#pragma once
#ifndef _WIN32
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <spawn.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>

extern char** environ;
namespace platform {
// Argument vectors are passed directly to the process, never through a shell.
class Process {
    pid_t pid_ = -1;
    int input_ = -1, output_ = -1;
public:
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;
    explicit Process(const std::vector<std::string>& args, bool merge_stderr = false) {
        if (args.empty()) throw std::runtime_error("empty process command");
        int in[2], out[2];
        if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, in))
            throw std::runtime_error("socketpair failed");
        if (pipe2(out, O_CLOEXEC)) {
            close(in[0]); close(in[1]);
            throw std::runtime_error("pipe failed");
        }
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, in[1], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&actions, out[1], STDOUT_FILENO);
        if (merge_stderr) posix_spawn_file_actions_adddup2(&actions, out[1], STDERR_FILENO);
        for (int fd : {in[0], in[1], out[0], out[1]})
            posix_spawn_file_actions_addclose(&actions, fd);
        std::vector<char*> argv;
        for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        int rc = posix_spawnp(&pid_, argv[0], &actions, nullptr, argv.data(), environ);
        posix_spawn_file_actions_destroy(&actions);
        close(in[1]); close(out[1]);
        if (rc) {
            close(in[0]); close(out[0]); pid_ = -1;
            throw std::runtime_error(args[0] + ": " + std::strerror(rc));
        }
        input_ = in[0]; output_ = out[0];
    }
    ~Process() {
        close_input();
        if (output_ >= 0) close(output_);
        if (pid_ > 0) {
            int status;
            if (waitpid(pid_, &status, WNOHANG) == 0) {
                kill(pid_, SIGKILL);
                while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
            }
        }
    }
    void close_input() { if (input_ >= 0) { close(input_); input_ = -1; } }
    void write(const std::string& data) {
        size_t offset = 0;
        while (offset < data.size()) {
            auto n = send(input_, data.data() + offset, data.size() - offset, MSG_NOSIGNAL);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("process input closed");
            offset += n;
        }
    }
    std::string read() {
        char buf[4096];
        ssize_t n;
        do { n = ::read(output_, buf, sizeof(buf)); } while (n < 0 && errno == EINTR);
        if (n < 0) throw std::runtime_error("process read failed");
        return std::string(buf, static_cast<size_t>(n));
    }
    int wait() {
        int status;
        pid_t rc;
        do { rc = waitpid(pid_, &status, 0); } while (rc < 0 && errno == EINTR);
        pid_ = -1;
        return rc > 0 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
};
inline int run(const std::vector<std::string>& args, std::string& output,
               const std::string& input = {}, bool merge_stderr = false) {
    Process p(args, merge_stderr);
    p.write(input); p.close_input(); output.clear();
    for (;;) { auto chunk = p.read(); if (chunk.empty()) break; output += chunk; }
    return p.wait();
}
}
#endif
