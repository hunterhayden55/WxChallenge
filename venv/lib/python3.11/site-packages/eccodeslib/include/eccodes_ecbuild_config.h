/*
 * (C) Copyright 2011- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#ifndef ECCODES_ecbuild_config_h
#define ECCODES_ecbuild_config_h

/* ecbuild info */

#ifndef ECBUILD_VERSION_STR
#define ECBUILD_VERSION_STR "3.13.0"
#endif
#ifndef ECBUILD_VERSION
#define ECBUILD_VERSION "3.13.0"
#endif
#ifndef ECBUILD_MACROS_DIR
#define ECBUILD_MACROS_DIR  "/opt/actions-runner/work/_work/python-develop-bundle/python-develop-bundle/ecbuild/cmake"
#endif

/* config info */

#define ECCODES_OS_NAME          "Darwin-22.5.0"
#define ECCODES_OS_BITS          64
#define ECCODES_OS_BITS_STR      "64"
#define ECCODES_OS_STR           "macosx.64"
#define ECCODES_OS_VERSION       "22.5.0"
#define ECCODES_SYS_PROCESSOR    "arm64"

#define ECCODES_BUILD_TIMESTAMP  "20251209080035"
#define ECCODES_BUILD_TYPE       "RelWithDebInfo"

#define ECCODES_C_COMPILER_ID      "AppleClang"
#define ECCODES_C_COMPILER_VERSION "14.0.3.14030022"

#define ECCODES_CXX_COMPILER_ID      "AppleClang"
#define ECCODES_CXX_COMPILER_VERSION "14.0.3.14030022"

#define ECCODES_C_COMPILER       "/Library/Developer/CommandLineTools/usr/bin/cc"
#define ECCODES_C_FLAGS          " -pipe -O2 -g -DNDEBUG"

#define ECCODES_CXX_COMPILER     "/Library/Developer/CommandLineTools/usr/bin/c++"
#define ECCODES_CXX_FLAGS        " -pipe -O2 -g -DNDEBUG"

/* Needed for finding per package config files */

#define ECCODES_INSTALL_DIR       "/tmp/eccodes/target/eccodes"
#define ECCODES_INSTALL_BIN_DIR   "/tmp/eccodes/target/eccodes/bin"
#define ECCODES_INSTALL_LIB_DIR   "/tmp/eccodes/target/eccodes/lib"
#define ECCODES_INSTALL_DATA_DIR  "/tmp/eccodes/target/eccodes/share/eccodes"

#define ECCODES_DEVELOPER_SRC_DIR "/opt/actions-runner/work/_work/python-develop-bundle/python-develop-bundle/src/eccodes"
#define ECCODES_DEVELOPER_BIN_DIR "/tmp/eccodes/build"

/* Fortran support */

#if 1

#define ECCODES_Fortran_COMPILER_ID      "GNU"
#define ECCODES_Fortran_COMPILER_VERSION "15.1.0"

#define ECCODES_Fortran_COMPILER "/opt/homebrew/bin/gfortran"
#define ECCODES_Fortran_FLAGS    " -O2 -g -DNDEBUG"

#endif

#endif /* ECCODES_ecbuild_config_h */
