/**
 * @file liborkh_nv_fatbin.h
 * @brief NVIDIA fat binary (fatbin) and CUDA ELF (cubin) parser.
 *
 * Extracts the GPU images embedded in a host executable / shared library
 * (sections .nv_fatbin and __nv_relfatbin), in a standalone .fatbin file or a
 * .cubin file, decompresses them (LZ4 or ZSTD fatbin compression) and parses
 * the static resource usage of every kernel of the cubins:
 * registers, shared / local / constant memory, stack, launch bounds, cluster
 * dimensions, parameters... from the .text.*, .nv.shared.*, .nv.constant0.*
 * sections and the .nv.info / .nv.info.* attribute sections (EIATTR_*).
 *
 * No dependency on the CUDA toolkit: works for every architecture from
 * Kepler (sm_30) to Blackwell (sm_100/sm_120), including the CUDA 12.8+
 * ELF ABI.
 */
#ifndef LIBORKH_NV_FATBIN_H
#define LIBORKH_NV_FATBIN_H

#include <stddef.h>
#include <stdint.h>

#define LIBORKH_NV_FATBIN_MAGIC         0xBA55ED50u
#define LIBORKH_NV_FATBIN_WRAPPER_MAGIC 0x466243B1u

#define LIBORKH_NV_KIND_PTX   1
#define LIBORKH_NV_KIND_ELF   2

#define LIBORKH_NV_FLAG_64BIT     0x0000000000000001ull
#define LIBORKH_NV_FLAG_DEBUG     0x0000000000000002ull
#define LIBORKH_NV_FLAG_LINUX     0x0000000000000010ull
#define LIBORKH_NV_FLAG_COMPRESS  0x0000000000002000ull

/** Raw attribute of a .nv.info section (kept for the JSON dump). */
typedef struct {
    uint8_t  format;     /**< EIFMT_* */
    uint8_t  attr;       /**< EIATTR_* */
    uint16_t size;       /**< Size of 'data' in bytes */
    const uint8_t* data; /**< Points into the image buffer */
} liborkh_nv_attr_t;

typedef struct {
    char*    name;                  /**< Mangled name */
    int      is_entry;              /**< 1 for __global__ kernels, 0 for device functions */
    uint32_t registers;
    uint32_t text_size;             /**< SASS code size in bytes */
    uint32_t shared_static;         /**< Static shared memory (bytes) */
    uint32_t local_size;            /**< .nv.local.* size (old toolchains) */
    uint32_t constant0_size;        /**< Constant bank 0 (parameters + driver data) */
    uint32_t param_size;            /**< Kernel parameters size (bytes) */
    uint32_t num_params;
    uint32_t frame_size;            /**< Local memory (stack frame) per thread */
    uint32_t min_stack_size;
    uint32_t max_stack_size;
    uint32_t crs_stack_size;
    uint32_t max_reg_count;         /**< __launch_bounds__/maxrregcount limit (0 if none) */
    uint32_t max_threads[3];        /**< __launch_bounds__ max threads (0 if none) */
    uint32_t req_ntid[3];           /**< .reqntid (0 if none) */
    uint32_t cluster_dim[3];        /**< __cluster_dims__ (sm_90+, 0 if none) */
    uint32_t max_cluster_rank;
    uint32_t num_barriers;
    uint32_t num_exits;
    uint32_t flags;                 /**< LIBORKH_NV_KFLAG_* */
    size_t   nattrs;
    liborkh_nv_attr_t* attrs;       /**< Per-kernel .nv.info.<name> attributes */
} liborkh_nv_kernel_t;

#define LIBORKH_NV_KFLAG_CTAIDZ_USED   0x01
#define LIBORKH_NV_KFLAG_WMMA_USED     0x02
#define LIBORKH_NV_KFLAG_COOP_GROUPS   0x04
#define LIBORKH_NV_KFLAG_ATOMIC_SYS    0x08
#define LIBORKH_NV_KFLAG_MBARRIER      0x10

typedef struct {
    uint32_t kind;                  /**< LIBORKH_NV_KIND_ELF or _PTX */
    uint32_t sm;                    /**< Real (ELF) or virtual (PTX) architecture, e.g. 90 */
    uint32_t virtual_sm;            /**< Virtual architecture of an ELF image (from e_flags) */
    uint32_t elf_abi_version;
    uint16_t ptx_major, ptx_minor;  /**< PTX ISA version (PTX images) */
    uint64_t flags;                 /**< Fatbin entry flags */
    int      compressed;            /**< 1 LZ4, 2 ZSTD, 0 none */
    char*    origin;                /**< Section the image comes from */
    char*    identifier;            /**< Object/module name stored in the fatbin (if any) */
    uint8_t* data;                  /**< Decompressed image (owned) */
    size_t   size;
    size_t   nkernels;
    liborkh_nv_kernel_t* kernels;
} liborkh_nv_image_t;

typedef struct {
    size_t count;
    liborkh_nv_image_t* images;
} liborkh_nv_image_list_t;

/**
 * Load every GPU image of 'path' (host ELF, .fatbin or .cubin).
 * @return 0 on success, -1 on error (message in 'err').
 */
int liborkh_nv_load_file(const char* path, liborkh_nv_image_list_t* out, char* err, size_t errlen);

/** Parse a fatbin buffer (one or more concatenated fatbins). */
int liborkh_nv_parse_fatbin(const uint8_t* buf, size_t size, const char* origin, liborkh_nv_image_list_t* out);

/** Parse a cubin buffer (kernels + resources). 'img->data/size' must be set. */
int liborkh_nv_parse_cubin(liborkh_nv_image_t* img);

void liborkh_nv_free_list(liborkh_nv_image_list_t* list);

/** Name of an EIATTR_* attribute ("EIATTR_0xNN" if unknown). */
const char* liborkh_nv_attr_name(uint8_t attr);

/** Decompress an LZ4 block. @return decompressed size or -1. */
long liborkh_nv_lz4_decompress(const uint8_t* src, size_t src_size, uint8_t* dst, size_t dst_capacity);

#endif /* LIBORKH_NV_FATBIN_H */
