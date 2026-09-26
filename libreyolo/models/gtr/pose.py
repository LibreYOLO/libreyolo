"""GTR pose: the GTR backbone and encoder with the DETRPose keypoint decoder.

Configuration source: Intellindust-AI-Lab/GTR (MIT), revision
782e737efe2e6437ac537fbdcee089673d3376c1, configs/pose/coco_pose_finetune.
Upstream ``GTRPoseTransformer`` has the same architecture and tensor names as
EdgeCrafter's ECPose decoder, so this reuses
:class:`~libreyolo.models.ec.decoder.ECPoseTransformer` with one change: GTR's
decoder layer keeps the keypoint position embedding on the attention value,
the residual path and the gate input (see :class:`GTRPoseDecoderLayer`).
"""

from torch import nn

from ..ec.decoder import ECPoseTransformer, PoseDeformableTransformerDecoderLayer
from .encoder import GTREncoder
from .nn import SIZE_CONFIGS
from .spatial import ViTAdapterSpatialSwiGLU

# (decoder layers, decoder feed-forward width) per published pose checkpoint.
POSE_SIZE_CONFIGS = {
    "s": (3, 512),
    "m": (4, 512),
    "l": (4, 1024),
    "x": (4, 2048),
}


class GTRPoseDecoderLayer(PoseDeformableTransformerDecoderLayer):
    """DETRPose layer with upstream GTR's position-embedding semantics.

    DETRPose adds the keypoint position embedding in place, so every later use
    of the tokens sees it. Upstream GTR writes that out of place and reassigns
    the result, which keeps it on the within-instance value and residual and on
    the gated cross-attention input. The EC layer only adds it to queries and
    keys, which changes GTR's outputs.
    """

    # GTR reassigns out of place in every mode, so unlike EdgeCrafter nothing
    # leaks into the previous layer's features.
    eval_pos_aliases_input = False

    def forward(
        self,
        tgt_pose,
        tgt_pose_query_pos,
        tgt_pose_reference_points,
        attn_mask=None,
        memory=None,
        memory_spatial_shapes=None,
    ):
        bs, nq, num_kpt, d_model = tgt_pose.shape

        tgt_pose = self.with_pos_embed(tgt_pose, tgt_pose_query_pos)
        tokens = tgt_pose.flatten(0, 1)
        within = self.within_attn(tokens, tokens, tokens)[0]
        tgt_pose = self.within_norm(
            tgt_pose + self.within_dropout(within.reshape(bs, nq, num_kpt, d_model))
        )

        tgt_pose = tgt_pose.transpose(1, 2).flatten(0, 1)
        across = self.across_attn(tgt_pose, tgt_pose, tgt_pose, attn_mask=attn_mask)[0]
        tgt_pose = self.across_norm(tgt_pose + self.across_dropout(across))
        tgt_pose = tgt_pose.reshape(bs, num_kpt, nq, d_model).transpose(1, 2)

        tgt_pose = self.with_pos_embed(tgt_pose, tgt_pose_query_pos)
        cross = self.cross_attn(
            tgt_pose.flatten(1, 2),
            tgt_pose_reference_points,
            memory,
            memory_spatial_shapes,
        ).reshape(bs, nq, num_kpt, d_model)
        tgt_pose = self.gateway(tgt_pose, self.dropout1(cross))
        return self.forward_ffn(tgt_pose)


class LibreGTRPoseModel(nn.Module):
    POSE_NUM_KEYPOINTS = 17
    # DETRPose's two-logit head; the person score is the last logit.
    POSE_NUM_CLASSES = 2

    def __init__(self, config="s", eval_spatial_size=(640, 640)):
        super().__init__()
        embed, heads, ratio, hidden, _ = SIZE_CONFIGS[config]
        layers, feedforward = POSE_SIZE_CONFIGS[config]
        self.backbone = ViTAdapterSpatialSwiGLU(
            embed_dim=embed,
            num_heads=heads,
            ffn_ratio=ratio,
            interaction_indexes=[3, 7, 11],
            multi_layer_same_res=True,
            skip_weights_warning=True,
            eval_spatial_size=eval_spatial_size,
        )
        self.encoder = GTREncoder(
            in_channels=[embed] * 3,
            hidden_dim=hidden,
            eval_spatial_size=eval_spatial_size,
        )
        self.decoder = ECPoseTransformer(
            hidden_dim=hidden,
            num_queries=60,
            num_decoder_layers=layers,
            dim_feedforward=feedforward,
            num_feature_levels=3,
            dec_n_points=4,
            num_keypoints=self.POSE_NUM_KEYPOINTS,
            num_classes=self.POSE_NUM_CLASSES,
            feat_strides=(8, 16, 32),
            eval_spatial_size=list(eval_spatial_size) if eval_spatial_size else None,
            reg_max=32,
            reg_scale=4.0,
        )
        self.decoder.decoder.layers = nn.ModuleList(
            GTRPoseDecoderLayer(
                d_model=hidden,
                d_ffn=feedforward,
                dropout=0.0,
                activation="relu",
                n_levels=3,
                n_heads=8,
                n_points=4,
            )
            for _ in range(layers)
        )

    def forward(self, x, targets=None):
        if x.shape[-2] != x.shape[-1] or x.shape[-1] % 32:
            raise ValueError("GTR expects a square input with side divisible by 32")
        # ``samples`` gives the denoising group the image size during training.
        return self.decoder(self.encoder(self.backbone(x)), targets=targets, samples=x)

    def deploy(self):
        # Only the decoder has a lossless deploy form; the backbone keeps the
        # portable graph (see LibreGTRModel.deploy).
        self.eval()
        self.decoder.convert_to_deploy()
        return self


class GTRPoseExportWrapper(nn.Module):
    """Tracing wrapper returning ``(pred_logits, pred_keypoints)``."""

    def __init__(self, model: LibreGTRPoseModel):
        super().__init__()
        self.model = model.deploy()

    def forward(self, x):
        out = self.model(x)
        keypoints = out["pred_keypoints"]
        if keypoints.dim() == 4:
            keypoints = keypoints.flatten(-2)
        return out["pred_logits"], keypoints
