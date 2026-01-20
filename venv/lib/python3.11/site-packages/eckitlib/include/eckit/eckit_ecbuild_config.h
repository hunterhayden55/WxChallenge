/*
 * (C) Copyright 2011- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#ifndef ECKIT_ecbuild_config_h
#define ECKIT_ecbuild_config_h

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

#define ECKIT_OS_NAME          "Darwin-22.5.0"
#define ECKIT_OS_BITS          64
#define ECKIT_OS_BITS_STR      "64"
#define ECKIT_OS_STR           "macosx.64"
#define ECKIT_OS_VERSION       "22.5.0"
#define ECKIT_SYS_PROCESSOR    "arm64"

#define ECKIT_BUILD_TIMESTAMP  "20251209075829"
#define ECKIT_BUILD_TYPE       "RelWithDebInfo"

#define ECKIT_C_COMPILER_ID      "AppleClang"
#define ECKIT_C_COMPILER_VERSION "14.0.3.14030022"

#define ECKIT_CXX_COMPILER_ID      "AppleClang"
#define ECKIT_CXX_COMPILER_VERSION "14.0.3.14030022"

#define ECKIT_C_COMPILER       "/Library/Developer/CommandLineTools/usr/bin/cc"
#define ECKIT_C_FLAGS          " -pipe -O2 -g -DNDEBUG"

#define ECKIT_CXX_COMPILER     "/Library/Developer/CommandLineTools/usr/bin/c++"
#define ECKIT_CXX_FLAGS        " -pipe -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-sign-compare -O2 -g -DNDEBUG"

/* Needed for finding per package config files */

#define ECKIT_INSTALL_DIR       "/tmp/eckit/target/eckit"
#define ECKIT_INSTALL_BIN_DIR   "/tmp/eckit/target/eckit/bin"
#define ECKIT_INSTALL_LIB_DIR   "/tmp/eckit/target/eckit/lib"
#define ECKIT_INSTALL_DATA_DIR  "/tmp/eckit/target/eckit/share/eckit"

#define ECKIT_DEVELOPER_SRC_DIR "/opt/actions-runner/work/_work/python-develop-bundle/python-develop-bundle/src/eckit"
#define ECKIT_DEVELOPER_BIN_DIR "/tmp/eckit/build"

/* Fortran support */

#if 0

#define ECKIT_Fortran_COMPILER_ID      ""
#define ECKIT_Fortran_COMPILER_VERSION ""

#define ECKIT_Fortran_COMPILER ""
#define ECKIT_Fortran_FLAGS    ""

#endif

#endif /* ECKIT_ecbuild_config_h */
