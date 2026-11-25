/**
 * @file liborkh_nv_fatbin.c
 * @brief NVIDIA fatbin / cubin parser (see liborkh_nv_fatbin.h).
 *
 * Fatbin container (reverse engineered, stable since CUDA 4):
 *   header : u32 magic (0xBA55ED50) | u16 version | u16 header_size | u64 fat_size
 *   entries: u16 kind (1 PTX, 2 ELF) | u16 version | u32 header_size | u64 payload_size
 *            u32 compressed_size | u32 unk | u16 minor | u16 major | u32 arch
 *            u32 name_offset | u32 name_len | u64 flags | u64 zero | u64 uncompressed_size
 *            ... payload at entry + header_size
 *   Compressed payloads (flag 0x2000) use the LZ4 block format, or ZSTD
 *   (recognised by its frame magic) with recent toolkits.
 *
 * CUDA ELF (cubin):
 *   .text.<fn>         sh_info[31:24] = registers, sh_size = code size
 *   .nv.shared.<fn>    static shared memory
 *   .nv.constant0.<fn> constant bank 0 (parameters)
 *   .nv.info           global attributes indexed by symbol (REGCOUNT, FRAME_SIZE, ...)
 *   .nv.info.<fn>      per kernel attributes (KPARAM_INFO, REQNTID, MAX_THREADS, ...)
 *   attribute = u8 format | u8 attr | value (format: 1 none, 2 u8, 3 u16, 4 u16 size + data)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <elf.h>
#include <zstd.h>

#include "liborkh_nv_fatbin.h"

#ifndef EM_CUDA
#define EM_CUDA 190
#endif
#define STO_CUDA_ENTRY 0x10

enum { EIFMT_NVAL = 1, EIFMT_BVAL = 2, EIFMT_HVAL = 3, EIFMT_SVAL = 4 };

/* EIATTR identifiers (from the CUDA ELF tools, see CuAssembler / nvdisasm-re). */
static const char* eiattr_names[] = {
    [0x00] = "EIATTR_ERROR",                      [0x01] = "EIATTR_PAD",
    [0x02] = "EIATTR_IMAGE_SLOT",                 [0x03] = "EIATTR_JUMPTABLE_RELOCS",
    [0x04] = "EIATTR_CTAIDZ_USED",                [0x05] = "EIATTR_MAX_THREADS",
    [0x06] = "EIATTR_IMAGE_OFFSET",               [0x07] = "EIATTR_IMAGE_SIZE",
    [0x08] = "EIATTR_TEXTURE_NORMALIZED",         [0x09] = "EIATTR_SAMPLER_INIT",
    [0x0a] = "EIATTR_PARAM_CBANK",                [0x0b] = "EIATTR_SMEM_PARAM_OFFSETS",
    [0x0c] = "EIATTR_CBANK_PARAM_OFFSETS",        [0x0d] = "EIATTR_SYNC_STACK",
    [0x0e] = "EIATTR_TEXID_SAMPID_MAP",           [0x0f] = "EIATTR_EXTERNS",
    [0x10] = "EIATTR_REQNTID",                    [0x11] = "EIATTR_FRAME_SIZE",
    [0x12] = "EIATTR_MIN_STACK_SIZE",             [0x13] = "EIATTR_SAMPLER_FORCE_UNNORMALIZED",
    [0x14] = "EIATTR_BINDLESS_IMAGE_OFFSETS",     [0x15] = "EIATTR_BINDLESS_TEXTURE_BANK",
    [0x16] = "EIATTR_BINDLESS_SURFACE_BANK",      [0x17] = "EIATTR_KPARAM_INFO",
    [0x18] = "EIATTR_SMEM_PARAM_SIZE",            [0x19] = "EIATTR_CBANK_PARAM_SIZE",
    [0x1a] = "EIATTR_QUERY_NUMATTRIB",            [0x1b] = "EIATTR_MAXREG_COUNT",
    [0x1c] = "EIATTR_EXIT_INSTR_OFFSETS",         [0x1d] = "EIATTR_S2RCTAID_INSTR_OFFSETS",
    [0x1e] = "EIATTR_CRS_STACK_SIZE",             [0x1f] = "EIATTR_NEED_CNP_WRAPPER",
    [0x20] = "EIATTR_NEED_CNP_PATCH",             [0x21] = "EIATTR_EXPLICIT_CACHING",
    [0x22] = "EIATTR_ISTYPEP_USED",               [0x23] = "EIATTR_MAX_STACK_SIZE",
    [0x24] = "EIATTR_SUQ_USED",                   [0x25] = "EIATTR_LD_CACHEMOD_INSTR_OFFSETS",
    [0x26] = "EIATTR_LOAD_CACHE_REQUEST",         [0x27] = "EIATTR_ATOM_SYS_INSTR_OFFSETS",
    [0x28] = "EIATTR_COOP_GROUP_INSTR_OFFSETS",   [0x29] = "EIATTR_COOP_GROUP_MASK_REGIDS",
    [0x2a] = "EIATTR_SW1850030_WAR",              [0x2b] = "EIATTR_WMMA_USED",
    [0x2c] = "EIATTR_HAS_PRE_V10_OBJECT",         [0x2d] = "EIATTR_ATOMF16_EMUL_INSTR_OFFSETS",
    [0x2e] = "EIATTR_ATOM16_EMUL_INSTR_REG_MAP",  [0x2f] = "EIATTR_REGCOUNT",
    [0x30] = "EIATTR_SW2393858_WAR",              [0x31] = "EIATTR_INT_WARP_WIDE_INSTR_OFFSETS",
    [0x32] = "EIATTR_SHARED_SCRATCH",             [0x33] = "EIATTR_STATISTICS",
    [0x34] = "EIATTR_INDIRECT_BRANCH_TARGETS",    [0x35] = "EIATTR_SW2861232_WAR",
    [0x36] = "EIATTR_SW_WAR",                     [0x37] = "EIATTR_CUDA_API_VERSION",
    [0x38] = "EIATTR_NUM_MBARRIERS",              [0x39] = "EIATTR_MBARRIER_INSTR_OFFSETS",
    [0x3a] = "EIATTR_COROUTINE_RESUME_ID_OFFSETS",[0x3b] = "EIATTR_SAM_REGION_STACK_SIZE",
    [0x3c] = "EIATTR_PER_REG_TARGET_PERF_STATS",  [0x3d] = "EIATTR_CTA_PER_CLUSTER",
    [0x3e] = "EIATTR_EXPLICIT_CLUSTER",           [0x3f] = "EIATTR_MAX_CLUSTER_RANK",
    [0x40] = "EIATTR_INSTR_REG_MAP",              [0x41] = "EIATTR_RESERVED_SMEM_USED",
    [0x42] = "EIATTR_RESERVED_SMEM_0_SIZE",       [0x43] = "EIATTR_UCODE_SECTION_DATA",
    [0x44] = "EIATTR_UNUSED_LOAD_BYTE_OFFSET",    [0x45] = "EIATTR_KPARAM_INFO_V2",
    [0x46] = "EIATTR_SYSCALL_OFFSETS",            [0x47] = "EIATTR_SW_WAR_MEMBAR_SYS_INSTR_OFFSETS",
    [0x48] = "EIATTR_GRAPHICS_GLOBAL_CBANK",      [0x49] = "EIATTR_SHADER_TYPE",
    [0x4a] = "EIATTR_VRC_CTA_INIT_COUNT",         [0x4b] = "EIATTR_TOOLS_PATCH_FUNC",
    [0x4c] = "EIATTR_NUM_BARRIERS",               [0x4d] = "EIATTR_TEXMODE_INDEPENDENT",
    /* Identified by aligning the output of 'cuobjdump -elf' (CUDA 13.1). */
    [0x50] = "EIATTR_SPARSE_MMA_MASK",            [0x5f] = "EIATTR_MERCURY_ISA_VERSION",
};

enum {
    A_CTAIDZ_USED = 0x04, A_MAX_THREADS = 0x05, A_PARAM_CBANK = 0x0a, A_REQNTID = 0x10,
    A_FRAME_SIZE = 0x11, A_MIN_STACK_SIZE = 0x12, A_KPARAM_INFO = 0x17, A_CBANK_PARAM_SIZE = 0x19,
    A_MAXREG_COUNT = 0x1b, A_EXIT_INSTR_OFFSETS = 0x1c, A_CRS_STACK_SIZE = 0x1e,
    A_MAX_STACK_SIZE = 0x23, A_ATOM_SYS_INSTR_OFFSETS = 0x27, A_COOP_GROUP_INSTR_OFFSETS = 0x28,
    A_WMMA_USED = 0x2b, A_REGCOUNT = 0x2f, A_NUM_MBARRIERS = 0x38, A_CTA_PER_CLUSTER = 0x3d,
    A_EXPLICIT_CLUSTER = 0x3e, A_MAX_CLUSTER_RANK = 0x3f, A_KPARAM_INFO_V2 = 0x45, A_NUM_BARRIERS = 0x4c,
};

const char* liborkh_nv_attr_name(uint8_t attr)
{
    static char buf[16][16];
    static int slot = 0;
    if (attr < sizeof(eiattr_names) / sizeof(eiattr_names[0]) && eiattr_names[attr]) return eiattr_names[attr];
    char* b = buf[slot++ & 15];
    snprintf(b, 16, "EIATTR_0x%02x", attr);
    return b;
}

static inline uint16_t rd16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t rd32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t rd64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }

/* ------------------------------------------------------------------------- */
/* LZ4 block decompression                                                    */
/* ------------------------------------------------------------------------- */

long liborkh_nv_lz4_decompress(const uint8_t* src, size_t src_size, uint8_t* dst, size_t dst_cap)
{
    size_t ip = 0, op = 0;
    while (ip < src_size) {
        uint8_t token = src[ip++];
        size_t lit = token >> 4;
        if (lit == 15) {
            uint8_t b;
            do { if (ip >= src_size) return -1; b = src[ip++]; lit += b; } while (b == 255);
        }
        if (ip + lit > src_size || op + lit > dst_cap) return -1;
        memcpy(dst + op, src + ip, lit);
        ip += lit; op += lit;
        if (ip >= src_size) break;           /* last sequence: literals only */
        if (ip + 2 > src_size) return -1;
        size_t off = (size_t)src[ip] | ((size_t)src[ip + 1] << 8);
        ip += 2;
        if (off == 0 || off > op) return -1;
        size_t mlen = token & 0x0f;
        if (mlen == 15) {
            uint8_t b;
            do { if (ip >= src_size) return -1; b = src[ip++]; mlen += b; } while (b == 255);
        }
        mlen += 4;
        if (op + mlen > dst_cap) return -1;
        for (size_t i = 0; i < mlen; i++, op++) dst[op] = dst[op - off]; /* overlapping copy */
    }
    return (long)op;
}

/* ------------------------------------------------------------------------- */
/* Cubin parsing                                                              */
/* ------------------------------------------------------------------------- */

static int known_sm(uint32_t sm)
{
    static const uint32_t sms[] = { 20, 21, 30, 32, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87,
                                    88, 89, 90, 100, 101, 103, 110, 120, 121, 0 };
    for (int i = 0; sms[i]; i++) if (sms[i] == sm) return 1;
    return 0;
}

static liborkh_nv_kernel_t* find_or_add_kernel(liborkh_nv_image_t* img, const char* name, size_t* cap)
{
    for (size_t i = 0; i < img->nkernels; i++)
        if (strcmp(img->kernels[i].name, name) == 0) return &img->kernels[i];
    if (img->nkernels == *cap) {
        size_t ncap = *cap ? *cap * 2 : 16;
        liborkh_nv_kernel_t* k = realloc(img->kernels, ncap * sizeof(*k));
        if (!k) return NULL;
        img->kernels = k;
        *cap = ncap;
    }
    liborkh_nv_kernel_t* k = &img->kernels[img->nkernels++];
    memset(k, 0, sizeof(*k));
    k->name = strdup(name);
    return k;
}

static void add_attr(liborkh_nv_kernel_t* k, uint8_t fmt, uint8_t attr, uint16_t size, const uint8_t* data)
{
    liborkh_nv_attr_t* a = realloc(k->attrs, (k->nattrs + 1) * sizeof(*a));
    if (!a) return;
    k->attrs = a;
    a[k->nattrs].format = fmt;
    a[k->nattrs].attr = attr;
    a[k->nattrs].size = size;
    a[k->nattrs].data = data;
    k->nattrs++;
}

/* Iterate over the attributes of a .nv.info section. */
typedef void (*attr_cb_t)(uint8_t fmt, uint8_t attr, uint16_t size, const uint8_t* data, void* user);

static void for_each_attr(const uint8_t* p, size_t size, attr_cb_t cb, void* user)
{
    size_t off = 0;
    while (off + 2 <= size) {
        uint8_t fmt = p[off], attr = p[off + 1];
        off += 2;
        uint16_t vsize = 0;
        const uint8_t* data = p + off;
        /* Every attribute starts with a 4 byte header: format, attribute and a
         * 16-bit field (unused for NVAL, the value for BVAL/HVAL, the size of
         * the data that follows for SVAL). */
        if (fmt != EIFMT_SVAL && fmt >= EIFMT_NVAL && fmt <= EIFMT_HVAL) {
            if (off + 2 > size) return;
            vsize = (fmt == EIFMT_NVAL) ? 0 : (fmt == EIFMT_BVAL ? 1 : 2);
            cb(fmt, attr, vsize, data, user);
            off += 2;
            continue;
        }
        switch (fmt) {
            case EIFMT_SVAL:
                if (off + 2 > size) return;
                vsize = rd16(p + off);
                off += 2;
                data = p + off;
                break;
            default: return; /* unknown format: stop */
        }
        if (off + vsize > size) return;
        cb(fmt, attr, vsize, data, user);
        off += vsize;
    }
}

typedef struct {
    liborkh_nv_image_t* img;
    const Elf64_Sym* syms;
    size_t nsyms;
    const char* strtab;
    size_t strtab_size;
    size_t* cap;
} global_ctx_t;

static const char* sym_name(global_ctx_t* g, uint32_t idx)
{
    if (!g->syms || idx >= g->nsyms) return NULL;
    uint32_t n = g->syms[idx].st_name;
    if (!g->strtab || n >= g->strtab_size) return NULL;
    return g->strtab + n;
}

static void global_attr_cb(uint8_t fmt, uint8_t attr, uint16_t size, const uint8_t* d, void* user)
{
    global_ctx_t* g = (global_ctx_t*)user;
    if (fmt != EIFMT_SVAL || size < 8) return;
    const char* name = sym_name(g, rd32(d));
    if (!name || !*name) return;
    liborkh_nv_kernel_t* k = find_or_add_kernel(g->img, name, g->cap);
    if (!k) return;
    uint32_t v = rd32(d + 4);
    switch (attr) {
        case A_REGCOUNT:       if (!k->registers) k->registers = v; break;
        case A_FRAME_SIZE:     k->frame_size = v; break;
        case A_MIN_STACK_SIZE: k->min_stack_size = v; break;
        case A_MAX_STACK_SIZE: k->max_stack_size = v; break;
        case A_CRS_STACK_SIZE: k->crs_stack_size = v; break;
        default: break;
    }
}

static void kernel_attr_cb(uint8_t fmt, uint8_t attr, uint16_t size, const uint8_t* d, void* user)
{
    liborkh_nv_kernel_t* k = (liborkh_nv_kernel_t*)user;
    add_attr(k, fmt, attr, size, d);
    switch (attr) {
        case A_MAX_THREADS:
            if (size >= 12) { k->max_threads[0] = rd32(d); k->max_threads[1] = rd32(d + 4); k->max_threads[2] = rd32(d + 8); }
            break;
        case A_REQNTID:
            if (size >= 12) { k->req_ntid[0] = rd32(d); k->req_ntid[1] = rd32(d + 4); k->req_ntid[2] = rd32(d + 8); }
            break;
        case A_CTA_PER_CLUSTER:
        case A_EXPLICIT_CLUSTER:
            if (size >= 12) { k->cluster_dim[0] = rd32(d); k->cluster_dim[1] = rd32(d + 4); k->cluster_dim[2] = rd32(d + 8); }
            break;
        case A_MAX_CLUSTER_RANK:
            if (size >= 4) k->max_cluster_rank = rd32(d);
            break;
        case A_MAXREG_COUNT:
            if (fmt == EIFMT_HVAL) k->max_reg_count = rd16(d);
            else if (size >= 4) k->max_reg_count = rd32(d);
            break;
        case A_PARAM_CBANK:
            /* u32 symbol index, u16 offset, u16 size */
            if (size >= 8) k->param_size = rd16(d + 6);
            break;
        case A_CBANK_PARAM_SIZE:
            if (fmt == EIFMT_HVAL) k->param_size = rd16(d);
            break;
        case A_KPARAM_INFO:
            k->num_params++;
            break;
        case A_KPARAM_INFO_V2:
            k->num_params++;
            break;
        case A_EXIT_INSTR_OFFSETS:
            k->num_exits += size / 4;
            break;
        case A_CRS_STACK_SIZE:
            if (size >= 4) k->crs_stack_size = rd32(d);
            break;
        case A_NUM_BARRIERS:
            if (fmt == EIFMT_HVAL) k->num_barriers = rd16(d);
            else if (size >= 4) k->num_barriers = rd32(d);
            break;
        case A_CTAIDZ_USED:             k->flags |= LIBORKH_NV_KFLAG_CTAIDZ_USED; break;
        case A_WMMA_USED:               k->flags |= LIBORKH_NV_KFLAG_WMMA_USED; break;
        case A_COOP_GROUP_INSTR_OFFSETS:k->flags |= LIBORKH_NV_KFLAG_COOP_GROUPS; break;
        case A_ATOM_SYS_INSTR_OFFSETS:  k->flags |= LIBORKH_NV_KFLAG_ATOMIC_SYS; break;
        case A_NUM_MBARRIERS:           k->flags |= LIBORKH_NV_KFLAG_MBARRIER; break;
        default: break;
    }
}

int liborkh_nv_parse_cubin(liborkh_nv_image_t* img)
{
    const uint8_t* b = img->data;
    size_t size = img->size;
    if (!b || size < sizeof(Elf64_Ehdr) || memcmp(b, ELFMAG, SELFMAG) != 0) return -1;
    if (b[EI_CLASS] != ELFCLASS64) return -1;

    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)b;
    if (eh->e_machine != EM_CUDA) return -1;
    if (eh->e_shoff == 0 || eh->e_shoff + (uint64_t)eh->e_shnum * sizeof(Elf64_Shdr) > size) return -1;

    img->elf_abi_version = b[EI_ABIVERSION];
    /* Architecture encoded in e_flags: bits [7:0] (ABI <= 7) or [15:8] (ABI >= 8). */
    uint32_t sm_lo = eh->e_flags & 0xff, sm_hi = (eh->e_flags >> 8) & 0xff, vsm = (eh->e_flags >> 16) & 0xff;
    uint32_t sm = (img->elf_abi_version >= 8) ? sm_hi : sm_lo;
    if (!known_sm(sm)) sm = known_sm(sm_lo) ? sm_lo : (known_sm(sm_hi) ? sm_hi : sm);
    if (!img->sm) img->sm = sm;
    img->virtual_sm = vsm;

    const Elf64_Shdr* sh = (const Elf64_Shdr*)(b + eh->e_shoff);
    if (eh->e_shstrndx >= eh->e_shnum) return -1;
    const Elf64_Shdr* shstr = &sh[eh->e_shstrndx];
    if (shstr->sh_offset + shstr->sh_size > size) return -1;
    const char* names = (const char*)(b + shstr->sh_offset);

    global_ctx_t g = { .img = img };
    size_t cap = img->nkernels;
    g.cap = &cap;

    /* Symbol table */
    for (int i = 0; i < eh->e_shnum; i++) {
        if (sh[i].sh_type != SHT_SYMTAB) continue;
        if (sh[i].sh_offset + sh[i].sh_size > size || sh[i].sh_link >= eh->e_shnum) continue;
        g.syms = (const Elf64_Sym*)(b + sh[i].sh_offset);
        g.nsyms = sh[i].sh_size / sizeof(Elf64_Sym);
        const Elf64_Shdr* st = &sh[sh[i].sh_link];
        if (st->sh_offset + st->sh_size <= size) {
            g.strtab = (const char*)(b + st->sh_offset);
            g.strtab_size = st->sh_size;
        }
        break;
    }

    /* Functions (.text.<name>) */
    for (int i = 0; i < eh->e_shnum; i++) {
        if (sh[i].sh_name >= shstr->sh_size) continue;
        const char* sname = names + sh[i].sh_name;
        if (strncmp(sname, ".text.", 6) != 0) continue;
        liborkh_nv_kernel_t* k = find_or_add_kernel(img, sname + 6, &cap);
        if (!k) return -1;
        k->registers = (uint32_t)(sh[i].sh_info >> 24);
        k->text_size = (uint32_t)sh[i].sh_size;
        /* Entry point if the function symbol has STO_CUDA_ENTRY */
        uint32_t symidx = (uint32_t)(sh[i].sh_info & 0xffffff);
        if (g.syms && symidx < g.nsyms) k->is_entry = (g.syms[symidx].st_other & STO_CUDA_ENTRY) ? 1 : 0;
    }
    if (g.syms) {
        for (size_t s = 0; s < g.nsyms; s++) {
            if (ELF64_ST_TYPE(g.syms[s].st_info) != STT_FUNC || !(g.syms[s].st_other & STO_CUDA_ENTRY)) continue;
            const char* n = sym_name(&g, (uint32_t)s);
            if (!n) continue;
            liborkh_nv_kernel_t* k = find_or_add_kernel(img, n, &cap);
            if (k) k->is_entry = 1;
        }
    }

    /* Resources and attributes */
    for (int i = 0; i < eh->e_shnum; i++) {
        if (sh[i].sh_name >= shstr->sh_size) continue;
        const char* sname = names + sh[i].sh_name;
        const uint8_t* data = (sh[i].sh_type != SHT_NOBITS && sh[i].sh_offset + sh[i].sh_size <= size)
                              ? b + sh[i].sh_offset : NULL;
        if (strncmp(sname, ".nv.shared.", 11) == 0) {
            liborkh_nv_kernel_t* k = find_or_add_kernel(img, sname + 11, &cap);
            if (k) k->shared_static = (uint32_t)sh[i].sh_size;
        } else if (strncmp(sname, ".nv.local.", 10) == 0) {
            liborkh_nv_kernel_t* k = find_or_add_kernel(img, sname + 10, &cap);
            if (k) k->local_size = (uint32_t)sh[i].sh_size;
        } else if (strncmp(sname, ".nv.constant0.", 14) == 0) {
            liborkh_nv_kernel_t* k = find_or_add_kernel(img, sname + 14, &cap);
            if (k) k->constant0_size = (uint32_t)sh[i].sh_size;
        } else if (strcmp(sname, ".nv.info") == 0 && data) {
            for_each_attr(data, sh[i].sh_size, global_attr_cb, &g);
        } else if (strncmp(sname, ".nv.info.", 9) == 0 && data) {
            liborkh_nv_kernel_t* k = find_or_add_kernel(img, sname + 9, &cap);
            if (k) for_each_attr(data, sh[i].sh_size, kernel_attr_cb, k);
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Fatbin parsing                                                             */
/* ------------------------------------------------------------------------- */

static int push_image(liborkh_nv_image_list_t* out, liborkh_nv_image_t* img)
{
    liborkh_nv_image_t* n = realloc(out->images, (out->count + 1) * sizeof(*n));
    if (!n) return -1;
    out->images = n;
    out->images[out->count++] = *img;
    return 0;
}

static int parse_entry(const uint8_t* e, size_t avail, const char* origin, liborkh_nv_image_list_t* out, size_t* consumed)
{
    if (avail < 64) return -1;
    uint16_t kind        = rd16(e);
    uint32_t header_size = rd32(e + 4);
    uint64_t payload     = rd64(e + 8);
    uint32_t comp_size   = rd32(e + 16);
    uint16_t minor       = rd16(e + 24);
    uint16_t major       = rd16(e + 26);
    uint32_t arch        = rd32(e + 28);
    uint32_t name_off    = rd32(e + 32);
    uint32_t name_len    = rd32(e + 36);
    uint64_t flags       = rd64(e + 40);
    uint64_t uncomp      = (header_size >= 64) ? rd64(e + 56) : 0;

    if (header_size < 64 || header_size > avail || payload > avail - header_size) return -1;
    *consumed = header_size + payload;

    liborkh_nv_image_t img;
    memset(&img, 0, sizeof(img));
    img.kind = kind;
    img.sm = arch;
    img.flags = flags;
    img.ptx_major = major;
    img.ptx_minor = minor;
    img.origin = strdup(origin ? origin : "");
    if (name_off && name_len && name_off < header_size && name_off + name_len <= header_size) {
        img.identifier = (char*)calloc(1, name_len + 1);
        if (img.identifier) memcpy(img.identifier, e + name_off, name_len);
    }

    const uint8_t* p = e + header_size;
    size_t psize = (size_t)payload;

    if ((flags & LIBORKH_NV_FLAG_COMPRESS) || (psize >= 4 && rd32(p) == 0xFD2FB528u)) {
        size_t csize = comp_size ? comp_size : psize;
        if (csize > psize) csize = psize;
        if (psize >= 4 && rd32(p) == 0xFD2FB528u) {
            unsigned long long fsize = ZSTD_getFrameContentSize(p, csize);
            size_t cap = (fsize != ZSTD_CONTENTSIZE_ERROR && fsize != ZSTD_CONTENTSIZE_UNKNOWN) ? (size_t)fsize : (size_t)uncomp;
            if (cap == 0) cap = csize * 8;
            img.data = (uint8_t*)malloc(cap + 1);
            size_t r = img.data ? ZSTD_decompress(img.data, cap, p, csize) : 0;
            if (!img.data || ZSTD_isError(r)) { free(img.data); img.data = NULL; }
            else img.size = r;
            img.compressed = 2;
        } else {
            size_t cap = uncomp ? (size_t)uncomp : csize * 8;
            img.data = (uint8_t*)malloc(cap + 16);
            long r = img.data ? liborkh_nv_lz4_decompress(p, csize, img.data, cap + 16) : -1;
            if (r < 0) { free(img.data); img.data = NULL; }
            else img.size = (size_t)r;
            img.compressed = 1;
        }
        if (!img.data) {
            /* Keep the entry (architecture information) even if it cannot be decoded. */
            img.size = 0;
        }
    } else {
        img.data = (uint8_t*)malloc(psize + 1);
        if (img.data) { memcpy(img.data, p, psize); img.size = psize; img.data[psize] = 0; }
    }

    if (img.kind == LIBORKH_NV_KIND_ELF && img.data) {
        liborkh_nv_parse_cubin(&img);
    } else if (img.kind == LIBORKH_NV_KIND_PTX && img.data) {
        /* PTX: list the entry points (.entry name) for JIT compiled kernels. */
        size_t cap = 0;
        img.data[img.size] = 0; /* every allocation above keeps one spare byte */
        const char* s = (const char*)img.data;
        const char* end = s + img.size;
        while ((s = strstr(s, ".entry")) && s < end) {
            s += 6;
            while (s < end && (*s == ' ' || *s == '\t')) s++;
            const char* n0 = s;
            while (s < end && *s != '(' && *s != ' ' && *s != '\n' && *s != '\r' && *s != '\t') s++;
            if (s > n0) {
                char tmp[1024];
                size_t l = (size_t)(s - n0) < sizeof(tmp) - 1 ? (size_t)(s - n0) : sizeof(tmp) - 1;
                memcpy(tmp, n0, l); tmp[l] = 0;
                liborkh_nv_kernel_t* k = find_or_add_kernel(&img, tmp, &cap);
                if (k) k->is_entry = 1;
            }
        }
    }
    return push_image(out, &img);
}

int liborkh_nv_parse_fatbin(const uint8_t* buf, size_t size, const char* origin, liborkh_nv_image_list_t* out)
{
    size_t off = 0;
    int found = 0;
    while (off + 16 <= size) {
        if (rd32(buf + off) != LIBORKH_NV_FATBIN_MAGIC) { off += 4; continue; }
        uint16_t hsize = rd16(buf + off + 6);
        uint64_t fsize = rd64(buf + off + 8);
        if (hsize < 16 || off + hsize + fsize > size) { off += 4; continue; }
        size_t eoff = off + hsize, eend = off + hsize + fsize;
        while (eoff + 64 <= eend) {
            size_t consumed = 0;
            if (parse_entry(buf + eoff, eend - eoff, origin, out, &consumed) != 0 || consumed == 0) break;
            eoff += consumed;
            found++;
        }
        off = eend;
        off = (off + 7) & ~(size_t)7;
    }
    return found;
}

/* ------------------------------------------------------------------------- */
/* Files                                                                      */
/* ------------------------------------------------------------------------- */

static uint8_t* read_file(const char* path, size_t* size)
{
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    uint8_t* b = (uint8_t*)malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); b = NULL; }
    fclose(f);
    if (b) *size = (size_t)n;
    return b;
}

int liborkh_nv_load_file(const char* path, liborkh_nv_image_list_t* out, char* err, size_t errlen)
{
    out->count = 0;
    out->images = NULL;
    size_t size = 0;
    uint8_t* b = read_file(path, &size);
    if (!b) { snprintf(err, errlen, "cannot read '%s'", path); return -1; }

    int rc = 0;
    if (size >= 4 && rd32(b) == LIBORKH_NV_FATBIN_MAGIC) {
        /* standalone fatbin */
        liborkh_nv_parse_fatbin(b, size, "fatbin", out);
    } else if (size >= sizeof(Elf64_Ehdr) && memcmp(b, ELFMAG, SELFMAG) == 0 && b[EI_CLASS] == ELFCLASS64) {
        const Elf64_Ehdr* eh = (const Elf64_Ehdr*)b;
        if (eh->e_machine == EM_CUDA) {
            /* standalone cubin */
            liborkh_nv_image_t img;
            memset(&img, 0, sizeof(img));
            img.kind = LIBORKH_NV_KIND_ELF;
            img.origin = strdup("cubin");
            img.data = b;
            img.size = size;
            b = NULL;
            liborkh_nv_parse_cubin(&img);
            push_image(out, &img);
        } else if (eh->e_shoff && eh->e_shoff + (uint64_t)eh->e_shnum * sizeof(Elf64_Shdr) <= size
                   && eh->e_shstrndx < eh->e_shnum) {
            const Elf64_Shdr* sh = (const Elf64_Shdr*)(b + eh->e_shoff);
            const Elf64_Shdr* shstr = &sh[eh->e_shstrndx];
            const char* names = (shstr->sh_offset + shstr->sh_size <= size) ? (const char*)(b + shstr->sh_offset) : NULL;
            for (int i = 0; names && i < eh->e_shnum; i++) {
                if (sh[i].sh_name >= shstr->sh_size || sh[i].sh_type == SHT_NOBITS) continue;
                const char* sname = names + sh[i].sh_name;
                if (strcmp(sname, ".nv_fatbin") != 0 && strcmp(sname, "__nv_relfatbin") != 0) continue;
                if (sh[i].sh_offset + sh[i].sh_size > size) continue;
                liborkh_nv_parse_fatbin(b + sh[i].sh_offset, sh[i].sh_size, sname, out);
            }
        } else {
            snprintf(err, errlen, "invalid ELF file '%s'", path);
            rc = -1;
        }
    } else {
        snprintf(err, errlen, "'%s' is neither an ELF, a cubin nor a fatbin", path);
        rc = -1;
    }
    free(b);
    return rc;
}

void liborkh_nv_free_list(liborkh_nv_image_list_t* list)
{
    if (!list) return;
    for (size_t i = 0; i < list->count; i++) {
        liborkh_nv_image_t* img = &list->images[i];
        for (size_t k = 0; k < img->nkernels; k++) {
            free(img->kernels[k].name);
            free(img->kernels[k].attrs);
        }
        free(img->kernels);
        free(img->data);
        free(img->origin);
        free(img->identifier);
    }
    free(list->images);
    list->images = NULL;
    list->count = 0;
}
