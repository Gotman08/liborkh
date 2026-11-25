/* nv_inspect: dump the GPU images and the kernel resources of an NVIDIA binary
 * (host executable / shared library, .fatbin or .cubin). Used to validate the
 * liborkh NVIDIA parser against 'cuobjdump -res-usage'. */
#include <stdio.h>
#include <stdlib.h>
#include "liborkh_nv_fatbin.h"

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <host-binary|.fatbin|.cubin> [-a]\n", argv[0]);
        return 1;
    }
    int all = argc > 2;
    liborkh_nv_image_list_t list;
    char err[512];
    if (liborkh_nv_load_file(argv[1], &list, err, sizeof(err)) != 0) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    for (size_t i = 0; i < list.count; i++) {
        liborkh_nv_image_t* img = &list.images[i];
        printf("image %zu: %s sm_%u (virtual %u, abi %u) flags=0x%llx compression=%d size=%zu origin=%s id=%s kernels=%zu\n",
               i, img->kind == LIBORKH_NV_KIND_ELF ? "cubin" : "ptx", img->sm, img->virtual_sm, img->elf_abi_version,
               (unsigned long long)img->flags, img->compressed, img->size, img->origin,
               img->identifier ? img->identifier : "", img->nkernels);
        for (size_t k = 0; k < img->nkernels; k++) {
            liborkh_nv_kernel_t* kk = &img->kernels[k];
            if (!kk->is_entry && !all) continue;
            printf("  %s %s: REG:%u STACK:%u(frame %u, max %u) SHARED:%u LOCAL:%u CONSTANT[0]:%u PARAM:%u(%u) "
                   "MAXREG:%u MAXTHREADS:%u,%u,%u CLUSTER:%u,%u,%u CODE:%u\n",
                   kk->is_entry ? "kernel" : "func  ", kk->name, kk->registers, kk->min_stack_size, kk->frame_size,
                   kk->max_stack_size, kk->shared_static, kk->local_size, kk->constant0_size, kk->param_size,
                   kk->num_params, kk->max_reg_count, kk->max_threads[0], kk->max_threads[1], kk->max_threads[2],
                   kk->cluster_dim[0], kk->cluster_dim[1], kk->cluster_dim[2], kk->text_size);
            if (all)
                for (size_t a = 0; a < kk->nattrs; a++)
                    printf("      %s (fmt %u, %u bytes)\n", liborkh_nv_attr_name(kk->attrs[a].attr),
                           kk->attrs[a].format, kk->attrs[a].size);
        }
    }
    liborkh_nv_free_list(&list);
    return 0;
}
