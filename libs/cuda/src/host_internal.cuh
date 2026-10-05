#pragma once
// Назначение параметров, общих для функций ниже (tMax, nPts, h, ranges,
// writableVar, maxValue, preScaller, eps и т.д.), описано одним словарём
// в hostLibrary.cuh — здесь не дублируется.
#include "ucuda/cuda/hostLibrary.cuh"

// Writer'ы _config.csv: формат легаси-файлов живёт в data_export::legacy, а не
// здесь. Текст файлов от этого не изменился ни на байт — см. комментарий в
// data_export.h о том, почему опечатки и пробелы в нём сохранены.
#include "ucuda/storage/data_export.h"
#include <vector>

// numb[] -> double[] для legacy CSV-writer'ов.
//
// Они НАМЕРЕННО объявлены с const double* (см. комментарий над namespace legacy
// в data_export.h): при смене typedef numb на float вызов обязан не собраться,
// чтобы расширение типа было ВИДНО в коде, а не подменяло молча точность в
// файле. Растяжка сработала — вот это самое видимое расширение. Сами writer'ы
// и формат файлов не тронуты.
//
// Возвращает vector по значению; временный объект живёт до конца полного
// выражения, т.е. в течение всего вызова writer'а — .data() валиден.
// При numb == double это копия один-в-один (путь холодный: одна запись
// _config.csv на прогон, единицы элементов).
//
// n — сколько элементов ЧИТАЕТ writer, а не размер массива у вызывающего:
// у ranges это 4 для двухосевых диаграмм и 2 для однооосевых
// (data_export.cpp: two_axes в write_lyap_config).
static std::vector<double> to_dbl(const numb* p, int n)
{
	if (p == nullptr || n <= 0) return std::vector<double>();
	return std::vector<double>(p, p + n);
}

// Путь для сохранения результирующих файлов
//#define OUT_FILE_PATH "C:\\Users\\KiShiVi\\Desktop\\mat.csv"
//#define OUT_FILE_PATH "C:\\CUDA\\mat.csv"

// Директива, объявление которой выводит в консоль отладочные сообщения
#define DEBUG

