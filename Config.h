#pragma once
#include<iostream>
#include<cstdio>
namespace Config {
	const int INNODE = 2;
	const int HIDENODE = 4;
	const int OUTNODE = 1;

	const double lr = 0.8;	// Ñ§Ï°ÂÊ£¬learning rate
	const double threshold = 1e-4;
	const int max_epoch = static_cast<int>(1e+6);
}