// part of this code is stolen from http://eric-yuan.me/cpp-read-mnist/

#pragma once
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;


int reverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void loadMnistImages(const string& filename, vector< vector< int > > &images)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		int number_of_images = 0;
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		int n_rows = 0;
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		int n_cols = 0;
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);

		images.resize(number_of_images);
		for (int i = 0; i < number_of_images; ++i)
		{
			images[i].resize(n_rows * n_cols);
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char pixel = 0;
					file.read((char*)&pixel, sizeof(pixel));
					images[i][n_rows * r + c] = (int)pixel;
				}
			}
		}
	}
}

void loadMnistLabels(const string& filename, vector< int > &labels)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		int number_of_items = 0;
		file.read((char*)&number_of_items, sizeof(number_of_items));
		number_of_items = reverseInt(number_of_items);

		labels.resize(number_of_items);
		for (int i = 0; i < number_of_items; ++i)
		{
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			labels[i] = (int)label;
		}
	}
}


