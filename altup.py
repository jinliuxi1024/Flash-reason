class Gemma3nTextAltUp(nn.Module):
    """Alternating Updates (AltUp)

    The AltUp module wraps transformer layers. The `predict` step modifies the
    input to the transformer layer, and the `correct` step propagates the output
    of the transformer layer to the sparsely updated dimensions.

    See more in the research paper:
    https://proceedings.neurips.cc/paper_files/paper/2023/file/f2059277ac6ce66e7e5543001afa8bb5-Paper-Conference.pdf
    """

    def __init__(self, config: Gemma3nTextConfig):a
        super().__init__()
        self.config = config
        self.correct_output_scale = nn.Parameter(torch.zeros(self.config.hidden_size))
        self.correction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs**2, bias=False)
        self.modality_router = nn.Linear(self.config.hidden_size, self.config.altup_num_inputs, bias=False)
        self.router_norm = Gemma3nRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.register_buffer("router_input_scale", torch.tensor(self.config.hidden_size**-1.0), persistent=False)

    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
        router_inputs = self.router_norm(x) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        return torch.tanh(routed.float()).type_as(x)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predicts the output of a layer using a trainable map.

        Args:
            hidden_states: A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]`
                           derived by stacking the input embeddings.
        Returns:
            A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` containing the predictions.
        """
        # 1. 计算路由向量 (modalities)，这仅基于 "active" 的输入流
        modalities = self.compute_router_modalities(hidden_states[self.config.altup_active_idx])

        if self.training and self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight.data.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        # 2. 生成预测矩阵 (prediction matrix)
        #    - prediction_coefs 将路由向量映射到一个更大的向量
        #    - reshape 将其变为一个方阵，这个方阵决定了如何混合多个输入流
        all_coefs: torch.Tensor = (
            self.prediction_coefs(modalities)
            .reshape(*modalities.shape[:-1], self.config.altup_num_inputs, self.config.altup_num_inputs)
            .permute(0, 1, 3, 2)
        )

        # 3. 执行预测：使用预测矩阵混合所有输入流
        #    - 通过矩阵乘法，将所有输入流 (hidden_states) 线性组合，生成对下一层输入的预测
        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2)  # 恢复原始维度顺序

        # 4. 添加残差连接
        predictions += hidden_states
        return predictions.contiguous().type_as(hidden_states)

    def correct(self, predictions: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        """
        Corrects the predictions relative to the actual transformer block output.

        Args:
            predictions: The output from the `predict` step.
            activated: The actual output from the transformer layer (after attention and MLP).
        Returns:
            A 4D tensor with corrected predictions.
        """
        # 1. 计算路由向量，这次基于 transformer 块的真实输出
        modalities = self.compute_router_modalities(activated)
        
        # 2. 计算 "innovation"：即真实输出与预测输出之间的差异
        innovation = activated - predictions[self.config.altup_active_idx]
        innovation = innovation.repeat(self.config.altup_num_inputs, 1, 1, 1)

        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight.data.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        # 3. 计算修正系数 (correction coefficients)
        #    - correction_coefs 将路由向量映射到一个修正向量
        #    - 加 1.0 使其成为一个以 1 为中心的乘性更新
        all_coefs: torch.Tensor = self.correction_coefs(modalities) + 1.0
        all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)

        # 4. 应用修正：将 innovation 按系数缩放，并加回到预测值上
        #    - 这将 "active" 流中计算出的更新信息传播到 "inactive" 流中
        corrected = torch.mul(innovation, all_coefs)
        corrected += predictions
        return corrected.contiguous().type_as(activated)

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        """Scales the provided 3D tensor of shape [batch_size, num_tokens, hidden_size]."""
        return (corrected.type_as(self.correct_output_scale) * self.correct_output_scale).type_as(corrected)