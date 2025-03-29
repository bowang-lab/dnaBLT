import pandas as pd
import math


def total_parameters(
    layers, hidden_state, transformer: bool, feed_forward_multiplier=4
):
    """Calculate total parameters for a transformer model"""
    if transformer:
        return (
            4 * hidden_state**2 + 2 * hidden_state**2 * feed_forward_multiplier
        ) * layers
    else:
        return (layers // 10) * (
            4 * hidden_state**2 + 2 * hidden_state**2 * feed_forward_multiplier
        ) + (layers - layers // 10) * (
            4 * hidden_state**2 + 3 * hidden_state**2 * feed_forward_multiplier
        )


def glu_flops(layers, hidden_state, feed_forward_mult=4):
    """Calculate FLOPs for feed forward networks

    Lead multiple of 3 instead of 2 for SwiGLU.
    """
    return 3 * layers * 2 * hidden_state * feed_forward_mult * hidden_state


def qkvo_flops(layers, hidden_state, r):
    """Calculate FLOPs for query, key, value, and output projections"""
    return (r * 2 + 2) * 2 * layers * hidden_state**2


def attention_flops(layers, hidden_state, context):
    """Calculate FLOPs for attention mechanism"""
    return 4 * layers * hidden_state * (context + 1) // 2


def deembedding_flops(hidden_state, vocab=4):
    """Calculate FLOPs for deembedding operation"""
    return 2 * hidden_state * vocab


def projection_convs_flops(layers, hidden_state):
    return 18 * hidden_state * layers


def featurization_flops(layers, hidden_state):
    return 2 * hidden_state * layers


def convolutions_gates_flops(layers, hidden_state, context):
    return layers * (10 * math.log2(context) * hidden_state + 4 * hidden_state)


def transformer_flops(hidden_state, layers, context, vocab=None, feed_forward_mult=4):
    """Calculate total FLOPs for a transformer model"""

    return (
        glu_flops(layers, hidden_state, feed_forward_mult)
        + qkvo_flops(layers, hidden_state, 1)
        + attention_flops(layers, hidden_state, context)
        + deembedding_flops(hidden_state, vocab)
    )


def cross_attention_flops(layers, hidden_state, patch_size, ratio_q2k):
    """Calculate FLOPs for cross attention mechanism"""
    return attention_flops(layers, hidden_state, patch_size) + qkvo_flops(
        layers, hidden_state, ratio_q2k
    )


def striped_hyena_flops(D, layers, context, vocab, glu_mult=4, hyena_ratio=10):
    mha_glu_layers = layers // hyena_ratio  # use 10% mixing ratio optima
    hyena_layers = layers - mha_glu_layers
    mha_glu = (
        attention_flops(mha_glu_layers, D, context)
        + glu_flops(mha_glu_layers, D, glu_mult)
        + qkvo_flops(mha_glu_layers, D, 1)
    )
    hyena_glu = (
        qkvo_flops(hyena_layers, D, 1)
        + convolutions_gates_flops(hyena_layers, D, context)
        + projection_convs_flops(hyena_layers, D)
        + featurization_flops(hyena_layers, D)
        + glu_flops(hyena_layers, D, glu_mult)
    )
    return mha_glu + hyena_glu + deembedding_flops(D, vocab)


class BLTFLOPsCalculator:
    """
    A class for calculating FLOPs for the BLT architecture.

    """

    def __init__(self, global_model, encoder_model, decoder_model):
        # All models must accept parameters in this order: (hidden_state_g, layers_g, context, vocab, feed_forward_mult, n_heads)
        self.global_model = global_model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def forward_blt_flops(
        self,
        seq_len,
        patch_size,
        hidden_state_g,
        layers_g,
        hidden_state_e,
        layers_e,
        window_e,
        hidden_state_d,
        layers_d,
        window_d,
        ratio_patchdim2bytedim,
        feed_forward_mult,
        vocab=None,
    ):
        """Calculate forward pass FLOPs for a BLT model"""
        return (
            self.global_model(
                hidden_state_g,
                layers_g,
                seq_len / patch_size,
                0,
                feed_forward_mult,
            )
            / patch_size
            + self.encoder_model(
                hidden_state_e, layers_e, window_e, 0, feed_forward_mult
            )
            + self.decoder_model(
                hidden_state_d, layers_d, window_d, vocab, feed_forward_mult
            )
            + cross_attention_flops(
                layers_e,
                hidden_state_e,
                patch_size,
                patch_size / ratio_patchdim2bytedim,
            )
            * (ratio_patchdim2bytedim / patch_size)
            + cross_attention_flops(
                layers_d,
                hidden_state_d,
                ratio_patchdim2bytedim,
                ratio_patchdim2bytedim / patch_size,
            )
        )

    def total_flops(
        self,
        tokens,
        seq_len,
        patch_size,
        hidden_state_g,
        layers_g,
        hidden_state_e,
        layers_e,
        window_e,
        hidden_state_d,
        layers_d,
        window_d,
        ratio_patchdim2bytedim,
        vocab=None,
        feed_forward_mult=None,
    ):
        """Calculate total FLOPs for training a BLT model"""

        forward_flops = self.forward_blt_flops(
            seq_len,
            patch_size,
            hidden_state_g,
            layers_g,
            hidden_state_e,
            layers_e,
            window_e,
            hidden_state_d,
            layers_d,
            window_d,
            ratio_patchdim2bytedim,
            vocab,
            feed_forward_mult,
        )
        return 3 * tokens * forward_flops


class ExperimentGeneration:
    """
    A class for generating experiments for the BLT and Striped Hyena architectures.
    """

    def __init__(self, blt_flops_calculator: BLTFLOPsCalculator):
        self.blt_flops_calculator = blt_flops_calculator

    def run_scaling_experiments(
        self,
        patch_sizes,
        decoder_layers,
        n_heads_e,
        global_layers,
        n_heads_g,
        feed_forward_mult,
    ):
        # ratio exceeds 2 only 2B and above
        """
        Run scaling law experiments to find optimal model configurations given FLOP budgets.

        Args:
            flop_budgets (list): List of FLOP budgets to target
            token_ranges (dict): Dictionary mapping FLOP budget names to (min_tokens, max_tokens) tuples
            seq_len (int): Sequence length
            window_e (int): Window size for encoder
            window_d (int): Window size for decoder
            ratio_patchdim2bytedim (int): Ratio of patch dimension to byte dimension

        Returns:
            pandas.DataFrame: Results of scaling experiments
        """

        flop_budget_map = {
            "s": 8e18,
            "m": 2e19,
            "l": 4e19,
            "xl": 8e19,
        }

        parameter_ranges = {
            # taken from Evo isoflop parabolas for StripedHyena
            "s": (2e7, 1.2e8),
            "m": (4e7, 2e8),
            "l": (4e7, 5e8),
            "xl": (4e7, 1e9),
        }

        values = []

        for p in patch_sizes:
            for dl in decoder_layers:
                for gl in global_layers:
                    for eh in n_heads_e:
                        for gh in n_heads_g:
                            forward_flops = self.blt_flops_calculator.forward_blt_flops(
                                seq_len=8192,
                                patch_size=p,
                                hidden_state_g=gh * 64,
                                layers_g=gl,
                                hidden_state_e=eh * 64,
                                layers_e=1,
                                window_e=512,
                                hidden_state_d=eh * 64,
                                layers_d=dl,
                                window_d=512,
                                ratio_patchdim2bytedim=2,
                                feed_forward_mult=feed_forward_mult,
                                vocab=4,
                            )

                            global_params = total_parameters(
                                gl,
                                gh * 64,
                                transformer=False,
                                feed_forward_multiplier=feed_forward_mult,
                            )

                            encoder_params = total_parameters(
                                1,
                                eh * 64,
                                transformer=True,
                                feed_forward_multiplier=feed_forward_mult,
                            )

                            decoder_params = total_parameters(
                                dl,
                                eh * 64,
                                transformer=True,
                                feed_forward_multiplier=feed_forward_mult,
                            )

                            if (decoder_params < global_params * 0.08) or (decoder_params > global_params * 0.12):
                                continue

                            # Calculate tokens for each FLOP budget
                            for budget_name, flop_budget in flop_budget_map.items():
                                tokens = flop_budget / (3 * forward_flops)
                                min_parameters, max_parameters = parameter_ranges[
                                    budget_name
                                ]
                                if (
                                    min_parameters
                                    <= global_params + encoder_params + decoder_params
                                    <= max_parameters * p
                                ):
                                    values.append(
                                        (
                                            budget_name,
                                            p,
                                            tokens,
                                            encoder_params,
                                            global_params,
                                            decoder_params,
                                            gh * 64,
                                            eh * 64,
                                            gl,
                                            dl,
                                        )
                                    )

        # Create DataFrame from results
        return values

    def run_default_experiment(self):
        """Run a default experiment with predefined parameters"""
        global_layers = list(range(4, 16 + 1))
        decoder_layers = list(map(lambda x: x // 3.5, global_layers))
        n_heads_g = list(range(2, 12 + 1))
        n_heads_e = list(range(2, 12 + 1))

        # 64 * head = model_dim (stripedhyena)

        values = self.run_scaling_experiments(
            [1.5, 2, 4], decoder_layers, n_heads_e, global_layers, n_heads_g, 2.67
        )
        columns = [
            "FLOPs",
            "Patch size",
            "Tokens",
            "Encoder parameters",
            "Global model parameters",
            "Decoder parameters",
            "Global model dimension",
            "Encoder/Decoder dimension",
            "Global model layers",
            "Decoder layers",
        ]
        df = pd.DataFrame.from_records(values, columns=columns)
        df.to_csv("Compute_allocations2.csv", index=False)


if __name__ == "__main__":
    blt_flops_calculator = BLTFLOPsCalculator(
        striped_hyena_flops, transformer_flops, transformer_flops
    )
    experiment_generation = ExperimentGeneration(blt_flops_calculator)
    experiment_generation.run_default_experiment()
