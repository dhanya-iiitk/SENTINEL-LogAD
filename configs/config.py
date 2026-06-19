# ============================================================
# SENTINEL Configuration
# SEMantic-graph Transformer with Normalizing flow for Log anomaly
# ============================================================

class SENTINELConfig:

    # --- Dataset ---
    DATASETS        = ["HDFS", "BGL", "OpenStack"]
    DATA_DIR        = "/content/drive/MyDrive/SENTINEL/data"
    PARSED_DIR      = "/content/drive/MyDrive/SENTINEL/parsed"
    RESULTS_DIR     = "/content/drive/MyDrive/SENTINEL/results"
    LOG_DIR         = "/content/drive/MyDrive/SENTINEL/logs"

    # --- Drain Parser ---
    DRAIN_DEPTH     = 4
    DRAIN_ST        = 0.5          # similarity threshold
    DRAIN_MAX_CHILD = 100

    # Dataset-specific Drain settings
    DRAIN_SETTINGS  = {
        "HDFS":      {"depth": 4, "st": 0.5},
        "BGL":       {"depth": 4, "st": 0.5},
        "OpenStack": {"depth": 4, "st": 0.4},
    }

    # --- Session Construction ---
    SESSION_SETTINGS = {
        "HDFS":      {"method": "identifier",  "window": None, "stride": None},
        "BGL":       {"method": "sliding",     "window": 20,   "stride": 1},
        "OpenStack": {"method": "identifier",  "window": None, "stride": None},
    }
    MAX_SEQ_LEN     = 50           # pad / truncate sessions to this length
    MIN_SEQ_LEN     = 2            # discard sessions shorter than this

    # --- Train / Val / Test Split ---
    TRAIN_RATIO     = 0.80         # fraction of NORMAL sessions for training
    VAL_RATIO       = 0.10         # fraction of NORMAL sessions for validation
    # TEST = remaining normal + all anomalous (held out)

    # --- Development Mode ---
    DEV_MODE        = True         # set False for full dataset run
    DEV_SAMPLES     = 2000         # sessions to use in dev mode

    # --- Embedding ---
    VOCAB_SIZE      = 300          # max unique event types (padded if fewer)
    EMBED_DIM       = 128          # event embedding dimension
    PAD_IDX         = 0            # padding token index

    # --- Transformer Encoder ---
    TRANSFORMER_LAYERS  = 2
    ATTENTION_HEADS     = 4
    FFN_DIM             = 256
    DROPOUT             = 0.1
    MAX_SEQ_LEN_PE      = 512      # positional encoding max length

    # --- GNN Encoder ---
    GNN_HIDDEN      = 128
    GNN_LAYERS      = 2
    GNN_DROPOUT     = 0.1

    # --- Fusion ---
    FUSION_INPUT    = 256          # EMBED_DIM * 2 (transformer + GNN)
    FUSION_OUTPUT   = 256          # latent dimension fed to flow

    # --- Normalizing Flow (RealNVP) ---
    FLOW_LAYERS     = 4            # number of coupling layers
    FLOW_HIDDEN     = 256          # hidden units in coupling network
    LATENT_DIM      = 256          # must match FUSION_OUTPUT

    # --- Training ---
    BATCH_SIZE      = 64
    EPOCHS          = 50
    LEARNING_RATE   = 1e-3
    WEIGHT_DECAY    = 1e-5
    PATIENCE        = 10           # early stopping patience
    GRAD_CLIP       = 1.0
    USE_AMP         = True         # automatic mixed precision (fp16)

    # --- YOBDM Threshold ---
    # Youden-optimal threshold on validation anomaly scores
    THRESHOLD_METHOD = "youden"    # "youden" | "percentile" | "fixed"
    PERCENTILE       = 95          # used if THRESHOLD_METHOD = "percentile"

    # --- Explainability ---
    IG_STEPS        = 50           # integrated gradient steps
    TOP_K_EVENTS    = 5            # top-k events to report in explanation
    GROQ_MODEL      = "llama-3.1-70b-versatile"

    # --- Reproducibility ---
    SEED            = 42

    # --- Device ---
    DEVICE          = "cuda"       # auto-detected in notebooks


cfg = SENTINELConfig()
