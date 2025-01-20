//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/** @file
 *  @brief QNN LPAI Memory components
 */

#ifndef QNN_LPAI_MEM_H
#define QNN_LPAI_MEM_H

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
  QNN_LPAI_MEM_TYPE_DDR       = 1,
  QNN_LPAI_MEM_TYPE_LLC       = 2,
  QNN_LPAI_MEM_TYPE_TCM       = 3,
  QNN_LPAI_MEM_TYPE_UNDEFINED = 0x7fffffff
} QnnLpaiMem_MemType_t;

// clang-format on
#ifdef __cplusplus
}  // extern "C"
#endif

#endif // QNN_LPAI_MEM_H
