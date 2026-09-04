// SIH 26172 - EdgeWake KWS Model Header Wrapper
// Provides safe extern declarations for g_kws_model_data and length
// Prevents multiple-definition and duplicate-storage linker collisions.

#ifndef KWS_MODEL_H_
#define KWS_MODEL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Model payload size in bytes (29,200 bytes)
extern const unsigned int g_kws_model_data_len;

// 16-byte aligned FlatBuffer byte array
extern const unsigned char g_kws_model_data[];

#ifdef __cplusplus
}
#endif

#endif  // KWS_MODEL_H_
