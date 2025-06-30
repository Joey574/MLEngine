#pragma once
#include "globals.hpp"
#include "structs.hpp"
#include "jsondefs.hpp"
#include "yamldefs.hpp"
#include "yamldefaults.hpp"

#ifdef DEBUG
#define DEBUG_LOG(m) std::cout << __FILE__ << ":" << __LINE__ << " | " << m
#else
#define DEBUG_LOG(m)
#endif
