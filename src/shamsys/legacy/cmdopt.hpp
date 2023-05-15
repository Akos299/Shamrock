// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file cmdopt.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 * @version 0.1
 * @date 2022-03-21
 *
 * @copyright Copyright (c) 2022
 *
 */



#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "aliases.hpp"

namespace opts{

    bool has_option(const std::string_view &option_name);
    std::string_view get_option(const std::string_view &option_name);
    void register_opt(std::string name, std::optional<std::string> args,std::string description);
    void init(int argc, char *argv[]);
    void print_help();
    bool is_help_mode();


    int get_argc();
    char** get_argv();

}