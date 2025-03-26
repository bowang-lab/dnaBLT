"""

SCALING LAWS EXPERIMENTS

- Entropy threshold calibration -> get reasonable token counts
- For fixed FLOP budgets, train sweeping (model, data, entropy) triplets
    - Implement FLOPs based early stopping (plateau)

- Find equation relating optimal (model, data, entropy) triplet to FLOPs
- Train big boi given prospective FLOPs

8B model
-----

seq_len = 8192
layers_e = 1
n_heads_e = 20
hidden_state_e = 1280
layers_g = 32
n_heads_g = 32
hidden_state_g = 4096
layers_d = 6
n_heads_d = 20
hidden_state_d = 1280
ratio_patchdim2bytedim = 4
window_e = 512
window_d = 512

"""

"""
patch_size=[1, 2, 4, 6, 8]
tokens = [1e10, 2e10, 4e10, 6e10]
# BLT optima: 64 encoder/decoder dimension: encoder/decoder heads | 128 global dimension: global heads
decoder_layers = [5, 6, 7, 8, 9, 10]
n_heads_e = [12, 14, 16, 18, 20]
global_layers = [24, 26, 28, 30, 32, 34, 36, 38, 40]
n_heads_g = [10, 16, 20, 24, 32, 40]
"""

import pandas as pd

class TransformerFLOPsCalculator:
    """
    A class for calculating FLOPs and parameters for various transformer architectures,
    with a focus on BLT (Byte-Level Transformer) models and scaling law experiments.
    """
    
    def __init__(self, vocab_size=256, feed_forward_mult=4):
        """
        Initialize the TransformerFLOPsCalculator with default values.
        
        Args:
            vocab_size (int): Size of the vocabulary
            feed_forward_mult (int): Multiplier for feed forward network size
        """
        self.vocab_size = vocab_size
        self.feed_forward_mult = feed_forward_mult
        
    def feed_forward_flops(self, layers, hidden_state, feed_forward_mult=None):
        """Calculate FLOPs for feed forward networks"""
        if feed_forward_mult is None:
            feed_forward_mult = self.feed_forward_mult
        return 2 * layers * 2 * hidden_state * feed_forward_mult * hidden_state

    def qkvo_flops(self, layers, hidden_state, ratio_q2kv=1):
        """Calculate FLOPs for query, key, value, and output projections"""
        return (ratio_q2kv * 2 + 2) * 2 * layers * hidden_state ** 2

    def attention_flops(self, layers, hidden_state_key, n_heads, context):
        """Calculate FLOPs for attention mechanism"""
        return 4 * layers * hidden_state_key * n_heads * (context + 1) // 2

    def deembedding_flops(self, hidden_state, vocab=None):
        """Calculate FLOPs for deembedding operation"""
        if vocab is None:
            vocab = self.vocab_size
        return 2 * hidden_state * vocab

    def transformer_flops(self, hidden_state, layers, context, vocab=None, feed_forward_mult=None, n_heads=1, ratio_q2k=1):
        """Calculate total FLOPs for a transformer model"""
        if vocab is None:
            vocab = self.vocab_size
        if feed_forward_mult is None:
            feed_forward_mult = self.feed_forward_mult
            
        return (
            self.feed_forward_flops(layers, hidden_state, feed_forward_mult) + 
            self.qkvo_flops(layers, hidden_state, ratio_q2k) + 
            self.attention_flops(layers, hidden_state // n_heads, n_heads, context) + 
            self.deembedding_flops(hidden_state, vocab)
        )

    def cross_attention_flops(self, layers, hidden_state_key, n_heads, patch_size, ratio_q2k):
        """Calculate FLOPs for cross attention mechanism"""
        return (
            self.attention_flops(layers, hidden_state_key, n_heads, patch_size) + 
            self.qkvo_flops(layers, hidden_state_key * n_heads, ratio_q2k)
        )

    def forward_blt_flops(
        self, seq_len, patch_size, 
        hidden_state_g, layers_g, 
        hidden_state_e, layers_e, window_e, 
        hidden_state_d, layers_d, window_d, 
        ratio_patchdim2bytedim, 
        n_heads_e, n_heads_g, n_heads_d, 
        vocab=None, feed_forward_mult=None
    ):
        """Calculate forward pass FLOPs for a BLT model"""
        if vocab is None:
            vocab = self.vocab_size
        if feed_forward_mult is None:
            feed_forward_mult = self.feed_forward_mult
            
        return (
            self.transformer_flops(hidden_state_g, layers_g, seq_len/patch_size, 0, feed_forward_mult, n_heads_g)/patch_size +
            self.transformer_flops(hidden_state_e, layers_e, window_e, 0, feed_forward_mult, n_heads_e) +
            self.transformer_flops(hidden_state_d, layers_d, window_d, vocab, feed_forward_mult, n_heads_d) +
            self.cross_attention_flops(layers_e, hidden_state_e // n_heads_e, n_heads_e, patch_size, patch_size / ratio_patchdim2bytedim) * (ratio_patchdim2bytedim / patch_size) +
            self.cross_attention_flops(layers_d, hidden_state_d // n_heads_d, n_heads_d, ratio_patchdim2bytedim, ratio_patchdim2bytedim / patch_size)
        )

    def total_flops(
        self, tokens, seq_len, patch_size, 
        hidden_state_g, layers_g, 
        hidden_state_e, layers_e, window_e, 
        hidden_state_d, layers_d, window_d, 
        ratio_patchdim2bytedim, 
        n_heads_e, n_heads_g, n_heads_d, 
        vocab=None, feed_forward_mult=None
    ):
        """Calculate total FLOPs for training a BLT model"""
        if vocab is None:
            vocab = self.vocab_size
        if feed_forward_mult is None:
            feed_forward_mult = self.feed_forward_mult
            
        forward_flops = self.forward_blt_flops(
            seq_len, patch_size, 
            hidden_state_g, layers_g, 
            hidden_state_e, layers_e, window_e, 
            hidden_state_d, layers_d, window_d, 
            ratio_patchdim2bytedim, 
            n_heads_e, n_heads_g, n_heads_d, 
            vocab, feed_forward_mult
        )
        return 3 * tokens * forward_flops

    def total_parameters(self, layers, hidden_state):
        """Calculate total parameters for a transformer model"""
        return layers * (12 * hidden_state**2 + 13 * hidden_state)
    
    def run_scaling_experiments(self, flop_budgets, token_ranges, seq_len=8192, window_e=512, window_d=512, ratio_patchdim2bytedim=2):
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
        patch_size = [1, 2, 4, 6]
        decoder_layers = [5, 6, 7, 8, 9]
        n_heads_e = [2, 4, 6, 8, 10, 12, 14, 16]
        global_layers = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
        n_heads_g = [2, 4, 6, 8, 10, 12, 14, 16]
        
        flop_budget_map = {
            "s": flop_budgets[0],
            "m": flop_budgets[1],
            "l": flop_budgets[2],
            "xl": flop_budgets[3]
        }
        
        values = []
        
        for p in patch_size:
            for dl in decoder_layers:
                for gl in global_layers:
                    for eh in n_heads_e:
                        for gh in n_heads_g:
                            # Check if encoder parameters are at most 1% of global transformer parameters
                            if self.total_parameters(1, eh * 64) / self.total_parameters(gl, gh * 128) <= 0.01:
                                forward_flops = self.forward_blt_flops(
                                    seq_len=seq_len, 
                                    patch_size=p, 
                                    hidden_state_g=gh * 128, 
                                    layers_g=gl, 
                                    hidden_state_e=eh * 64, 
                                    layers_e=1, 
                                    window_e=window_e, 
                                    hidden_state_d=eh * 64, 
                                    layers_d=dl, 
                                    window_d=window_d, 
                                    ratio_patchdim2bytedim=ratio_patchdim2bytedim, 
                                    n_heads_e=eh, 
                                    n_heads_g=gh, 
                                    n_heads_d=eh
                                )
                                
                                # Calculate tokens for each FLOP budget
                                for budget_name, flop_budget in flop_budget_map.items():
                                    tokens = flop_budget / (3 * forward_flops)
                                    min_tokens, max_tokens = token_ranges[budget_name]
                                    
                                    if min_tokens <= tokens <= max_tokens:
                                        values.append((
                                            budget_name, 
                                            p, 
                                            tokens, 
                                            self.total_parameters(1, eh * 64), 
                                            self.total_parameters(gl, gh * 128), 
                                            self.total_parameters(dl, eh * 64), 
                                            gh * 128, 
                                            eh * 64, 
                                            gl, 
                                            dl
                                        ))
        
        # Create DataFrame from results
        columns = [
            "FLOPs", "Patch size", "Tokens", "Encoder parameters", 
            "Global transformer parameters", "Decoder parameters", 
            "Global transformer dimension", "Encoder/Decoder dimension", 
            "Global transformer layers", "Decoder layers"
        ]
        df = pd.DataFrame.from_records(values, columns=columns)
        return df
    
    def save_results(self, df, filename="Compute_allocations.csv"):
        """Save results to a CSV file"""
        df.to_csv(filename, index=False)
        return filename
    
    def run_default_experiment(self):
        """Run a default experiment with predefined parameters"""
        flop_budgets = [4e18, 8e18, 2e19, 4e19]
        token_ranges = {
            "s": (4e9, 7e10),
            "m": (6.66e9, 1.33e11),
            "l": (1.3e10, 6.66e11),
            "xl": (1.3e10, 6.66e11)
        }
        
        results = self.run_scaling_experiments(flop_budgets, token_ranges)
        self.save_results(results)
        return results


import math

class StripedHyenaFLOPsCalculator:
    """
    A class for calculating FLOPs for the Striped Hyena architecture.
    The total FLOP cost is a mixture of the Hyena‐GLU and MHA‐GLU costs,
    controlled by the mixing ratio λ (0 ≤ λ ≤ 1):
    
    FLOPS_StripedHyena = λ * FLOPS_Hyena-GLU + (1 − λ) * FLOPS_MHA-GLU
    
    The FLOP breakdown is as follows:

    ## MHA GLU FLOP Calculations


    - MHA projections:  
      6 * L * D²

    - MHA attention:  
      Instead of computing over a full L×L interaction, we assume
      a fixed context. The updated cost is:
      
      4 * (layers) * D * ((context + 1) / 2) = 2 * (layers) * D * (context + 1)

    - MHA output layer:  
      2 * L * D²

    - GLU:  
      6 * L * D * D_glu

    Summing these gives:

      FLOPS_MHA-GLU = 6LDD_glu + 8LD² + 2 * (layers) * D * (context + 1)

    ## Hyena-GLU FLOP Calculations

    - GLU (same as above):  
      6 * L * D * D_glu

    - Sequence Mixer – projections:  
      6 * L * D²

    - Sequence Mixer – convs on projections:  
      18 * L * D

    - Sequence Mixer – featurization:  
      S_hyena * L * D⁹

    - Sequence Mixer – convolution & gates:  
      10 * L * log₂(L) * D + 4 * L * D

    - Sequence Mixer – out layer:  
      2 * L * D²

    Combining these:
      - The D² parts: 6LD² + 2LD² = 8LD².
      - The LD parts: 18LD + 4LD + 10L·log₂(L)D = 22LD + 10L·log₂(L)D.

    Thus, the total for Hyena-GLU is:
    
      FLOPS_Hyena-GLU = 6LDD_glu + 8LD² + (22LD + 10L·log₂(L)D) + S_hyena * L * D⁹
    """

    def __init__(self, vocab_size=256, S_hyena=1.0):
        """
        Initialize the StripedHyenaFLOPsCalculator.
        
        Args:
            vocab_size (int): Vocabulary size (V) for embedding calculations.
            S_hyena (float): Scaling constant for the Hyena featurization FLOPs.
        """
        self.vocab_size = vocab_size
        self.S_hyena = S_hyena

    def mha_glu_flops(self, L, layers, D, D_glu, V=None, context=None):
        """
        Calculate FLOPs for the MHA-GLU branch.
        
        Args:
            L (int): The sequence length (or number of tokens).
            layers (int): Number of layers (for the attention attention cost).
            D (int): Model width.
            D_glu (int): Internal dimension for GLU.
            V (int): Vocabulary size. If None, uses the default.
            context (int): Context length for attention. If None, defaults to L.
            
        Returns:
            int: Estimated FLOPs for MHA-GLU.
        """
        if V is None:
            V = self.vocab_size
        if context is None:
            context = L

        # Embedding layers: 4 * L * D * V
        # embedding = 4 * L * D * V
        
        # MHA projections: 6 * L * D^2
        projections = 6 * L * (D ** 2)
        
        # MHA attention (updated to fixed context cost):
        # 4 * (layers) * D * ((context + 1)/2) = 2 * (layers) * D * (context + 1)
        attention = 2 * layers * D * (context + 1)
        
        # MHA output layer: 2 * L * D^2
        out_layer = 2 * L * (D ** 2)
        
        # GLU: 6 * L * D * D_glu
        glu = 6 * L * D * D_glu
        
        # Sum the components:
        # Note: projections + out_layer = (6 + 2) * L * D^2 = 8 * L * D^2
        total = glu + (projections + out_layer) + attention
        return total

    def hyena_glu_flops(self, L, layers, D, D_glu, V=None):
        """
        Calculate FLOPs for the Hyena-GLU branch.
        
        Args:
            L (int): The sequence length (or number of tokens).
            layers (int): Number of layers.
            D (int): Model width.
            D_glu (int): Internal dimension for GLU.
            V (int): Vocabulary size. If None, uses the default.
            
        Returns:
            int: Estimated FLOPs for Hyena-GLU.
        """
        if V is None:
            V = self.vocab_size

        # Embedding + GLU (same as in MHA-GLU):
        glu = 6 * L * D * D_glu
        
        # Sequence Mixer – projections: 6 * L * D^2
        mixer_proj = 6 * L * (D ** 2)
        
        # Sequence Mixer – convs on projections: 18 * L * D
        mixer_convs = 18 * L * D
        
        # Sequence Mixer – featurization: S_hyena * L * D
        mixer_feat = self.S_hyena * L * (D)
        
        # Sequence Mixer – convolution & gates: 10 * L * log₂(L) * D + 4 * L * D
        mixer_conv_gates = 10 * L * math.log2(L) * D + 4 * L * D
        
        # Sequence Mixer – out layer: 2 * L * D^2
        mixer_out = 2 * L * (D ** 2)
        
        # Combine the D^2 parts:
        # mixer_proj + mixer_out = 6LD^2 + 2LD^2 = 8LD^2.
        # LD parts: mixer_convs + mixer_conv_gates = 18LD + 4LD + 10L log₂(L)D.
        total_mixer = mixer_proj + mixer_convs + mixer_feat + mixer_conv_gates + mixer_out
        
        total = glu + total_mixer
        return total

    def striped_hyena_flops(self, lambda_val, L, layers, D, D_glu, V=None, context=None):
        """
        Calculate the total FLOPs for the Striped Hyena architecture.
        The FLOPs are a weighted sum of the Hyena-GLU and MHA-GLU FLOPs.
        
        Args:
            lambda_val (float): Mixing ratio for the Hyena branch (0 ≤ λ ≤ 1).
            L (int): Sequence length (or number of tokens).
            layers (int): Number of layers.
            D (int): Model width.
            D_glu (int): Internal dimension for GLU.
            V (int): Vocabulary size. If None, uses the default.
            context (int): Context length for attention. If None, defaults to L.
            
        Returns:
            int: Estimated total FLOPs for Striped Hyena.
        """
        if V is None:
            V = self.vocab_size
        if context is None:
            context = L

        mha_glu = self.mha_glu_flops(L, layers, D, D_glu, V, context)
        hyena_glu = self.hyena_glu_flops(L, layers, D, D_glu, V)
        total = lambda_val * hyena_glu + (1 - lambda_val) * mha_glu
        return total