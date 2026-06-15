#pragma once
#include "system_record.h"
#include <string>
#include <vector>

// Хранилище систем: папка с файлом JSON на каждую систему.
// Все операции работают с файлами на диске.
class SystemLibrary {
public:
    // dir — папка библиотеки (создаётся при необходимости).
    explicit SystemLibrary(std::string dir);

    // Список имён сохранённых систем (по именам файлов).
    std::vector<std::string> list() const;

    // Сохранить запись. Имя файла произ水одится от rec.name.
    // Если name пусто — генерируется "Untitled N". Возвращает итоговое имя.
    std::string save(const SystemRecord& rec);

    // Загрузить по имени. Бросает, если не найдено.
    SystemRecord load(const std::string& name) const;

    // Удалить по имени. Возвращает true, если папка удалена.
    bool remove(const std::string& name);

    // Переименовать: переносит папку old -> new со всем содержимым (система + НУ).
    // false, если old нет или new уже занято.
    bool rename(const std::string& old_name, const std::string& new_name);

    // Дублировать: загрузить, переименовать в "<name> (copy)", сохранить.
    // Возвращает имя копии.
    std::string duplicate(const std::string& name);

    // Существует ли система с таким именем.
    bool exists(const std::string& name) const;

    // --- сессии: подпапка sessions/ внутри папки системы ---
    // "_last" — зарезервированное имя авто-сохранённой последней сессии.
    // Именованные сессии сохраняются рядом и доступны для переключения.
    void save_session(const std::string& sysname, const std::string& session,
        const std::string& json) const;
    std::string load_session(const std::string& sysname,
        const std::string& session) const; // "" если нет
    bool has_session(const std::string& sysname, const std::string& session) const;
    bool remove_session(const std::string& sysname, const std::string& session) const;
    // Список именованных сессий (без "_last").
    std::vector<std::string> list_sessions(const std::string& sysname) const;

private:
    std::string dir_;
    std::string dir_for(const std::string& name) const;   // папка системы
    std::string path_for(const std::string& name) const;  // system.json внутри
    std::string sanitize(const std::string& name) const;
};

// --- JSON-сериализация записи (доступна и для тестов) ---
std::string record_to_json(const SystemRecord& rec);
SystemRecord record_from_json(const std::string& json);