# Picture pipeline, 4-fold cross-validation (finetune_v3)

Each fold's model on the drawings it never saw: its 3 held-out OPEN100 sheets, and LightUp (team).
Mean ± standard deviation over the folds. Connections are direct symbol-to-symbol lines.

## OPEN100 (held-out real sheets)

| Configuration | Items mAP | Items AP50 | Lines mAP | Conn. precision | Conn. recall | Conn. F1 | tank | pump | valve | instrumentation | general |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A model only | 0.689 ± 0.031 | 0.730 ± 0.038 | 0.281 ± 0.051 | 0.809 ± 0.044 | 0.671 ± 0.056 | 0.733 ± 0.048 | 0.365 ± 0.125 | 0.084 ± 0.102 | 0.959 ± 0.009 | 0.945 ± 0.038 | 0.738 ± 0.113 |
| B + tracer | 0.689 ± 0.031 | 0.730 ± 0.038 | 0.254 ± 0.043 | 0.807 ± 0.046 | 0.738 ± 0.051 | 0.771 ± 0.046 | 0.365 ± 0.125 | 0.084 ± 0.102 | 0.959 ± 0.009 | 0.945 ± 0.038 | 0.738 ± 0.113 |
| C + fixes | 0.694 ± 0.031 | 0.735 ± 0.037 | 0.287 ± 0.052 | 0.803 ± 0.045 | 0.674 ± 0.056 | 0.732 ± 0.050 | 0.367 ± 0.126 | 0.091 ± 0.114 | 0.967 ± 0.013 | 0.964 ± 0.026 | 0.738 ± 0.113 |
| D full pipeline | 0.694 ± 0.031 | 0.735 ± 0.037 | 0.260 ± 0.042 | 0.802 ± 0.047 | 0.742 ± 0.053 | 0.771 ± 0.048 | 0.367 ± 0.126 | 0.091 ± 0.114 | 0.967 ± 0.013 | 0.964 ± 0.026 | 0.738 ± 0.113 |

## LightUp (held-out team drawing)

| Configuration | Items mAP | Items AP50 | Lines mAP | Conn. precision | Conn. recall | Conn. F1 | tank | pump | valve | instrumentation | general |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A model only | 0.931 ± 0.066 | 0.940 ± 0.064 | 0.489 ± 0.039 | 0.815 ± 0.046 | 0.724 ± 0.045 | 0.766 ± 0.030 | 0.890 ± 0.033 | 0.917 ± 0.095 | 0.987 ± 0.005 | 0.973 ± 0.022 | 1.000 ± 0.000 |
| B + tracer | 0.931 ± 0.066 | 0.940 ± 0.064 | 0.518 ± 0.047 | 0.832 ± 0.042 | 0.840 ± 0.045 | 0.835 ± 0.016 | 0.890 ± 0.033 | 0.917 ± 0.095 | 0.987 ± 0.005 | 0.973 ± 0.022 | 1.000 ± 0.000 |
| C + fixes | 0.957 ± 0.070 | 0.961 ± 0.071 | 0.528 ± 0.041 | 0.811 ± 0.047 | 0.737 ± 0.040 | 0.771 ± 0.027 | 0.977 ± 0.009 | 1.000 ± 0.000 | 0.981 ± 0.024 | 0.989 ± 0.009 | 1.000 ± 0.000 |
| D full pipeline | 0.957 ± 0.070 | 0.961 ± 0.071 | 0.564 ± 0.072 | 0.832 ± 0.042 | 0.874 ± 0.059 | 0.850 ± 0.024 | 0.977 ± 0.009 | 1.000 ± 0.000 | 0.981 ± 0.024 | 0.989 ± 0.009 | 1.000 ± 0.000 |

Caveat: the refinement rules (C, D) and the tracer fixes were developed on the team drawings,
LightUp included - its gains there are optimistic. OPEN100 was not used for that.
