import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.core.xla_model as xm
import pickle
import jax
import os


from torch_xla.experimental.custom_kernel import (
    FlashAttention,
    jax_import_guard,
    trace_pallas,
)


def flash_attention(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    causal=False,
    q_segment_ids=None,  # [batch_size, q_seq_len]
    kv_segment_ids=None,  # [batch_size, kv_seq_len]
    sm_scale=1.0,
    *,
    ab=None,  # [batch_size, num_heads, q_seq_len, kv_seq_len]
    partition_spec=None,
    mesh=None,
):
    # TODO: support SPMD and Dynamo with segment_ids.
    return SPMDFlashAttention.apply(
        q,
        k,
        v,
        causal,
        q_segment_ids,
        kv_segment_ids,
        sm_scale,
        ab,
        partition_spec,
        mesh,
    )


class SPMDFlashAttention(FlashAttention):
    """
    This is a simplified wrapper on top of https://github.com/google/jax/blob/b2058d72b7e1693a41303d5411572aabf99b7981/jax/experimental/pallas/ops/tpu/flash_attention.py#L139
    where we only takes q, k, v and causal as input and set block_sizes for the users.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        causal,
        q_segment_ids,
        kv_segment_ids,
        sm_scale,
        ab,
        partition_spec,
        mesh,
    ):
        # Import JAX within the function such that we don't need to call the jax_import_guard()
        # in the global scope which could cause problems for xmp.spawn.
        jax_import_guard()
        import jax  # noqa: F401
        from jax.experimental.pallas.ops.tpu.flash_attention import (
            _flash_attention_impl,
        )

        ctx.causal = causal
        ctx.sm_scale = sm_scale
        ctx.partition_spec = partition_spec
        ctx.mesh = mesh
        ctx.q_full_shape = None
        ctx.kv_full_shape = None
        save_residuals = q.requires_grad or k.requires_grad or v.requires_grad

        # SPMD integration.
        # mark_sharding is in-placed, and therefore save the full q, k, v for the backward.
        full_q = q
        full_k = k
        full_v = v
        full_ab = ab

        if partition_spec is not None:
            ctx.q_full_shape = q.shape
            ctx.kv_full_shape = k.shape
            q = xs.enable_manual_sharding(q, partition_spec, mesh=mesh).global_tensor
            k = xs.enable_manual_sharding(k, partition_spec, mesh=mesh).global_tensor
            v = xs.enable_manual_sharding(v, partition_spec, mesh=mesh).global_tensor
            if ab:
                ab = xs.enable_manual_sharding(
                    ab, partition_spec, mesh=mesh
                ).global_tensor

        # It computes the shape and type of o, l, m.
        shapes = [q.shape]
        dtypes = [q.dtype]
        if save_residuals:
            res_shape = list(q.shape)
            res_shape[-1] = FlashAttention.MIN_BLOCK_SIZE
            for _ in range(2):
                shapes.append(res_shape)
                dtypes.append(torch.float32)

        with torch.no_grad():
            if (
                partition_spec is not None
                and q_segment_ids is not None
                and kv_segment_ids is not None
            ):
                # partition_spec is for q,k,v with shape [batch, num_head, seq_len, head_dim], segment id
                # is of shape [batch, seq_len], hence we need to tweak it a bit
                segment_id_partition_spec = (partition_spec[0], partition_spec[2])
                q_segment_ids = xs.enable_manual_sharding(
                    q_segment_ids, (partition_spec[0], partition_spec[2]), mesh=mesh
                ).global_tensor
                kv_segment_ids = xs.enable_manual_sharding(
                    kv_segment_ids, segment_id_partition_spec, mesh=mesh
                ).global_tensor
            segment_ids, q_segment_ids_fa, kv_segment_ids_fa = (
                FlashAttention.prepare_segment_ids(q_segment_ids, kv_segment_ids)
            )
            ctx.segment_ids = segment_ids

            # We can't directly use flash_attention as we need to override the save_residuals flag which returns
            # l and m that is needed for the backward. Then we lose all the shape checks.
            # TODO: replicate the shape checks on flash_attention.
            # Here we seperate the tracing and execution part just to support SegmentIds.
            payload, _ = trace_pallas(
                _flash_attention_impl,
                q,
                k,
                v,
                ab,
                segment_ids,
                save_residuals,
                causal,
                sm_scale,
                min(FlashAttention.DEFAULT_BLOCK_SIZES["block_b"], q.shape[0]),
                min(FlashAttention.DEFAULT_BLOCK_SIZES["block_q"], q.shape[2]),
                min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k_major"], k.shape[2]),
                min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k"], k.shape[2]),
                False,
                static_argnums=range(5, 13),
                use_cache=True,
            )

            args = [q, k, v]
            if ab is not None:
                args += [ab]
            if segment_ids is not None:
                args += [q_segment_ids_fa, kv_segment_ids_fa]
            o = torch_xla._XLAC._xla_tpu_custom_call(args, payload, shapes, dtypes)

            if not save_residuals:
                o = o[0]
                # SPMD integration
                if partition_spec is not None:
                    o = xs.disable_manual_sharding(
                        o, partition_spec, ctx.q_full_shape, mesh=mesh
                    ).global_tensor
                return o
            o, *aux = o
            l, m = (v[..., 0] for v in aux[-2:])  # noqa: E741

        # SPMD integration
        if partition_spec is not None:
            o = xs.disable_manual_sharding(
                o, partition_spec, ctx.q_full_shape, mesh=mesh
            ).global_tensor
            l = xs.disable_manual_sharding(  # noqa: E741
                l, partition_spec[0:3], ctx.q_full_shape[0:3], mesh=mesh
            ).global_tensor
            m = xs.disable_manual_sharding(
                m, partition_spec[0:3], ctx.q_full_shape[0:3], mesh=mesh
            ).global_tensor

        ctx.save_for_backward(
            full_q,
            full_k,
            full_v,
            o,
            l,
            m,
            q_segment_ids_fa,
            kv_segment_ids_fa,
            full_ab,
        )
        return o

    @staticmethod
    def backward(ctx, grad_output):
        from jax.experimental.pallas.ops.tpu.flash_attention import (
            _flash_attention_bwd_dq,
            _flash_attention_bwd_dkv,
        )

        q, k, v, o, l, m, q_segment_ids_fa, kv_segment_ids_fa, ab = (  # noqa: E741
            ctx.saved_tensors
        )
        causal = ctx.causal
        sm_scale = ctx.sm_scale
        partition_spec = ctx.partition_spec
        mesh = ctx.mesh
        q_full_shape = ctx.q_full_shape
        kv_full_shape = ctx.kv_full_shape
        segment_ids = ctx.segment_ids
        grad_q = grad_k = grad_v = grad_ab = None

        grad_i = torch.sum(
            o.to(torch.float32) * grad_output.to(torch.float32), axis=-1
        )  # [batch_size, num_heads, q_seq_len]

        expanded_l = l.unsqueeze(-1).expand(
            [-1 for _ in l.shape] + [FlashAttention.MIN_BLOCK_SIZE]
        )
        expanded_m = m.unsqueeze(-1).expand(
            [-1 for _ in m.shape] + [FlashAttention.MIN_BLOCK_SIZE]
        )
        expanded_grad_i = grad_i.unsqueeze(-1).expand(
            [-1 for _ in grad_i.shape] + [FlashAttention.MIN_BLOCK_SIZE]
        )

        # SPMD integration
        if partition_spec is not None:
            q = xs.enable_manual_sharding(q, partition_spec, mesh=mesh).global_tensor
            k = xs.enable_manual_sharding(k, partition_spec, mesh=mesh).global_tensor
            v = xs.enable_manual_sharding(v, partition_spec, mesh=mesh).global_tensor
            expanded_l = xs.enable_manual_sharding(
                expanded_l, partition_spec, mesh=mesh
            ).global_tensor
            expanded_m = xs.enable_manual_sharding(
                expanded_m, partition_spec, mesh=mesh
            ).global_tensor
            grad_output = xs.enable_manual_sharding(
                grad_output, partition_spec, mesh=mesh
            ).global_tensor
            expanded_grad_i = xs.enable_manual_sharding(
                expanded_grad_i, partition_spec, mesh=mesh
            ).global_tensor
            if ab:
                ab = xs.enable_manual_sharding(
                    ab, partition_spec, mesh=mesh
                ).global_tensor

        if ctx.needs_input_grad[0]:
            payload, _ = trace_pallas(
                _flash_attention_bwd_dq,
                q,
                k,
                v,
                ab,
                segment_ids,
                l,
                m,
                grad_output,
                grad_i,
                block_q_major=min(
                    FlashAttention.DEFAULT_BLOCK_SIZES["block_q_dq"], q.shape[2]
                ),
                block_k_major=min(
                    FlashAttention.DEFAULT_BLOCK_SIZES["block_k_major_dq"], k.shape[2]
                ),
                block_k=min(
                    FlashAttention.DEFAULT_BLOCK_SIZES["block_k_dq"], k.shape[2]
                ),
                sm_scale=sm_scale,
                causal=causal,
                mask_value=FlashAttention.DEFAULT_MASK_VALUE,
                debug=False,
                static_argnames=[
                    "block_q_major",
                    "block_k_major",
                    "block_k",
                    "sm_scale",
                    "causal",
                    "mask_value",
                    "debug",
                ],
                use_cache=True,
            )

            args = [q, k, v]
            if ab is not None:
                args += [ab]
            if segment_ids is not None:
                args += [q_segment_ids_fa, kv_segment_ids_fa]
            args += [expanded_l, expanded_m, grad_output, expanded_grad_i]

            outputs = [q]
            if ab is not None:
                outputs += [ab]
            grads = torch_xla._XLAC._xla_tpu_custom_call(
                args, payload, [i.shape for i in outputs], [i.dtype for i in outputs]
            )
            if ctx.needs_input_grad[0]:
                grad_q = grads[0]
            if ctx.needs_input_grad[-3]:
                grad_ab = grads[1]

        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            payload, _ = trace_pallas(
                _flash_attention_bwd_dkv,
                q,
                k,
                v,
                ab,
                segment_ids,
                l,
                m,
                grad_output,
                grad_i,
                block_q_major=min(
                    FlashAttention.DEFAULT_BLOCK_SIZES["block_q_major_dkv"], q.shape[2]
                ),
                block_k_major=min(
                    FlashAttention.DEFAULT_BLOCK_SIZES["block_k_major_dkv"], k.shape[2]
                ),
                block_k=min(
                    FlashAttention.DEFAULT_BLOCK_SIZES["block_k_dkv"], k.shape[2]
                ),
                block_q=min(
                    FlashAttention.DEFAULT_BLOCK_SIZES["block_q_dkv"], q.shape[2]
                ),
                sm_scale=sm_scale,
                causal=causal,
                mask_value=FlashAttention.DEFAULT_MASK_VALUE,
                debug=False,
                static_argnames=[
                    "block_q_major",
                    "block_k_major",
                    "block_k",
                    "block_q",
                    "sm_scale",
                    "causal",
                    "mask_value",
                    "debug",
                ],
                use_cache=True,
            )

            grads = torch_xla._XLAC._xla_tpu_custom_call(
                args, payload, [k.shape, v.shape], [k.dtype, v.dtype]
            )

        if ctx.needs_input_grad[1]:
            grad_k = grads[0]
        if ctx.needs_input_grad[2]:
            grad_v = grads[1]

        # SPMD integration
        if partition_spec is not None:
            grad_q = xs.disable_manual_sharding(
                grad_q, partition_spec, q_full_shape, mesh=mesh
            ).global_tensor
            grad_k = xs.disable_manual_sharding(
                grad_k, partition_spec, kv_full_shape, mesh=mesh
            ).global_tensor
            grad_v = xs.disable_manual_sharding(
                grad_v, partition_spec, kv_full_shape, mesh=mesh
            ).global_tensor

        return grad_q, grad_k, grad_v, None, None, None, None, grad_ab, None, None


if __name__ == "__main__":
    if len(os.sys.argv) < 2:
        print("Usage: python custom_kernel_spmd.py <use_spmd>")
        os.sys.exit(1)

    use_spmd = os.sys.argv[1]
    jax.config.update("jax_default_matmul_precision", "highest")
    mesh, attn_spec = None, None
    if use_spmd:
        import torch_xla.runtime as xr
        from torch_xla.distributed.spmd import Mesh
        import numpy as np

        xr.use_spmd()
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, 1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("data", "model", "sequence"))
        attn_spec = ("data", None, None, None)
    batch_size = 1000

    data_path = "data.pkl"
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            q, k, v, mask = pickle.load(f)
    else:
        q = torch.randn(batch_size, 2, 128, 4)
        k = torch.randn(batch_size, 2, 128, 4)
        v = torch.randn(batch_size, 2, 128, 4)
        mask = torch.rand(batch_size, 128)
        pickle.dump((q, k, v, mask), open(data_path, "wb"))

    q, k, v, mask = q.to("xla"), k.to("xla"), v.to("xla"), mask.to("xla")

    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    q_segment_indexes = torch.ones(
        batch_size, q.shape[2], device=q.device, dtype=torch.float32
    )

    grads_path = "grads.pkl"
    if os.path.exists(grads_path):
        print("loaded output")
        with open(grads_path, "rb") as f:
            o, q_grad, k_grad, v_grad = pickle.load(f)
        o, q_grad, k_grad, v_grad = (
            o.to("xla"),
            q_grad.to("xla"),
            k_grad.to("xla"),
            v_grad.to("xla"),
        )
    else:
        o = SPMDFlashAttention.apply(
            q, k, v, False, q_segment_indexes, mask, 1.0, attn_spec, mesh
        )
        print(f"created output with shape {o.shape}", flush=True)

        loss = o.sum()
        loss.backward()
        xm.mark_step()

        q_grad = q.grad
        k_grad = k.grad
        v_grad = v.grad

        o_cpu = o.cpu()

        with open("grads.pkl", "wb") as f:
            pickle.dump([o.cpu(), q_grad.cpu(), k_grad.cpu(), v_grad.cpu()], f)

    q.grad = None
    k.grad = None
    v.grad = None

    o2 = SPMDFlashAttention.apply(
        q, k, v, False, q_segment_indexes, mask, 1.0, attn_spec, mesh
    )
    loss = o2.sum()
    loss.backward()
    xm.mark_step()

    print(
        "comparing gradients (loaded / computed) to the gradients after computing the same again:"
    )
    for i, j in [(q_grad, q.grad), (k_grad, k.grad), (v_grad, v.grad)]:
        print(torch.allclose(i, j, rtol=1e-14))

    print("opposite")
    for i, j in [(q_grad, q.grad), (k_grad, k.grad), (v_grad, v.grad)]:
        print(torch.allclose(j, i, rtol=1e-14))
    print(f"comparing second output with shape: {o2.shape}", flush=True)
    print(torch.allclose(o, o2, rtol=1e-14))
