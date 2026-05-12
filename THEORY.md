# ACIT4530 Assignment C — Theory Notes

Companion document for `assignment_C.ipynb`. The goal here is to explain the *why* behind each
function call, what each parameter does, and how to read the results. The notebook does the
computation; this document explains the math.

**Scope.** The notebook implements **Data Exploration** and **CP / PARAFAC Tensor Factorization**.
The SVD section in this doc (§§ 3–5) is *background reading* — it explains how one would flatten
a 3-way tensor and apply SVD, why that's a reasonable baseline, and why CP is the more natural
choice for this dataset. The CP section (§ 6 onward) is what the notebook actually runs.

---

## 1. The data and the prediction task

The DBLP tensor `X` has shape **(471 authors × 366 conferences × 14 years)** spanning 1991–2004.
Each entry `X[i,j,k]` is the number of papers author *i* published at conference *j* in year *k*.
`Y` is the **2005** ground-truth matrix (471 × 366). We binarize `Y` (1 if any publication, 0
otherwise) and try to predict it from `X`.

This is a **temporal link-prediction** problem: given a sparse history, predict which
author–conference links appear next year. The framing follows Dunlavy, Kolda & Acar (2011).

---

## 2. Log-transform

```python
X_log = np.zeros_like(X, dtype=float)
mask = X > 0
X_log[mask] = np.log(X[mask]) + 1
```

**Why.** Publication counts are very skewed — a handful of authors have huge counts, most have
1–2. `log(x)+1` (applied only where `x > 0` so we don't blow up) compresses the tail without
collapsing the difference between "no paper" (0) and "one paper" (which maps to `log(1)+1 = 1`).
This is the transform the assignment explicitly prescribes.

**Why not `log(x+1)`?** Because the spec says `log(X) + 1` for non-zero entries only — keeping
zeros as exact zeros preserves the sparsity pattern, which matters for SVD reconstruction and
for the binary prediction task.

---

## 3. Flattening tensor → matrix for SVD *(background)*

SVD only works on matrices, so we collapse the year axis two different ways.

### Approach 1 — plain sum

```python
Z1 = X_log.sum(axis=2)            # shape (471, 366)
```

Every year contributes equally. Simple, no hyperparameter, but it forgets *when* a paper was
written. A 1991 paper counts the same as a 2004 paper when predicting 2005.

### Approach 2 — exponentially-weighted sum

`Z(i,j) = Σ_t (1 − θ)^(T − t) · X(i,j,t)`

```python
weights = (1 - theta) ** (T - 1 - np.arange(T))   # T = 14
Z2 = (X_log * weights[None, None, :]).sum(axis=2)
```

**What it assumes.** Recent years are more predictive of next year than old years. The weight
`(1 − θ)^(T − t)` decays exponentially as we go further into the past. With `T = 14`:

| `t` (year index) | weight when θ=0.25 | weight when θ=0.75 |
|---|---|---|
| 13 (year 2004, most recent) | `(0.75)^0 = 1.00` | `(0.25)^0 = 1.00` |
| 12 (2003) | 0.75 | 0.25 |
| 7  (1998) | 0.178 | 0.00024 |
| 0  (1991) | 0.024 | ≈10⁻⁸ |

**Reading θ.**
- θ → 0 ⇒ weights flatten toward 1, Z2 → plain sum (Approach 1).
- θ → 1 ⇒ weights collapse to "only the most recent year matters."
- Mid-range θ (≈0.25–0.5) gives a smooth recency bias — usually the sweet spot for predicting
  the *next* time step from a slow-moving process like academic publishing.

The choice of θ is a hyperparameter we tune on the prediction AUC.

---

## 4. Truncated SVD — `np.linalg.svd` + rank-K truncation *(background)*

```python
U, s, Vt = np.linalg.svd(Z, full_matrices=False)
Z_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

**Why this is the "best" rank-K approximation.** The Eckart–Young theorem: among all matrices
of rank ≤ K, `Z_k` minimizes both the Frobenius and spectral reconstruction error. SVD gives us
this optimum in closed form.

**`full_matrices=False`** — returns the **thin** SVD. For an `m × n` matrix with `m > n`, the
full `U` is `m × m` (mostly zeros) and the thin `U` is `m × n`. Same approximation, much less
memory.

**Reconstruction error.** Measured as `||Z − Z_k||_F`. By Eckart–Young, this equals
`sqrt(Σ_{i>K} s_i²)` — the sum of squared singular values we threw away. Always **monotonically
decreases** as K grows, and hits exactly 0 when K equals the matrix rank. So the question
"what's the best K" is *not* answered by reconstruction error alone — it'll always prefer the
biggest K. We need a downstream task (prediction AUC) to find the genuinely useful K.

### Picking K for prediction

Big K reconstructs the *training* matrix perfectly, including its noise. For predicting 2005
(which the model has never seen), some K in the middle is best — large enough to capture real
structure, small enough to avoid memorizing the past. We sweep `K ∈ {2, 10, 20, 50, 100, 300}`
and pick the K with the highest AUC on `Y`.

---

## 5. ROC curve & AUC — `sklearn.metrics`

```python
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(y_true_flat, scores_flat)
auc = roc_auc_score(y_true_flat, scores_flat)
```

**What we feed it.** `y_true_flat` is the flattened binarized `Y` (1 if author *i* published at
conference *j* in 2005, else 0). `scores_flat` is the flattened `Z_k` — we use raw
reconstruction values as the score, not thresholded predictions, so the ROC curve can sweep
every possible threshold.

**Reading AUC.**
- 0.5 = random guessing.
- 1.0 = perfect ranking (every positive scored above every negative).
- For sparse link prediction on real-world data, AUC in the **0.85–0.95** range is good; above
  0.95 is excellent.

**Why AUC and not accuracy?** `Y` is extremely imbalanced — only ~2% of entries are 1. A model
that always predicts 0 gets ~98% accuracy and is useless. AUC measures *ranking* quality, which
is what we want: "did the model rank true future links higher than non-links?"

---

## 6. CP / PARAFAC decomposition — `tensorly.decomposition.parafac`

```python
from tensorly.decomposition import parafac
weights, factors = parafac(X_log, rank=R, init='random', n_iter_max=500, tol=1e-7,
                           return_errors=False, random_state=seed)
```

**What CP does.** It factors a 3-way tensor `X` into a sum of `R` rank-one tensors:

```
X[i,j,k] ≈ Σ_{r=1..R}  λ_r · A[i,r] · B[j,r] · C[k,r]
```

So each component `r` is a triple of vectors — one per mode (authors, conferences, years) —
plus a weight `λ_r`. Reading a component means looking at all three vectors together: the
authors that load high on `A[:,r]`, the conferences that load high on `B[:,r]`, and the
temporal shape `C[:,r]`. That's a *theme* — a group of authors who publish at a group of
conferences with a particular time profile.

**Why CP and not Tucker?** Tucker decomposition uses a core tensor and is more flexible but
**non-unique**. CP is essentially unique under mild conditions (Kruskal's theorem), which is
what makes its components *interpretable*. That uniqueness is the whole reason CP is the
workhorse for exploratory tensor analysis.

### Key parameters

| Param | What it does | Why we use what we use |
|---|---|---|
| `rank` | Number of rank-one components R | The biggest knob. R too small ⇒ underfit. R too large ⇒ overfit, components start splitting/duplicating. We sweep R and watch fit + FMS. |
| `init='random'` | Each call starts from a different random factor matrix | CP is non-convex; the algorithm (ALS) only finds local optima. Multiple random starts is the standard defence. |
| `n_iter_max=500` | Cap on ALS iterations | Has to be high enough that convergence isn't truncated. 500 is comfortable for this tensor; we check the actual iteration count to see if we hit the cap. |
| `tol=1e-7` | Convergence threshold (relative change in error) | Tight enough that the solution is essentially stable, loose enough that we don't grind forever in numerical noise. |
| `return_errors=True` | Returns the per-iteration error history | Lets us read off both the final fit and the number of iterations actually taken — both required deliverables. |
| `random_state=seed` | Seeds the init | Reproducibility. We vary it across the 10–20 runs. |

**What ALS does.** Alternating Least Squares: hold two factor matrices fixed, solve for the
third by least squares; rotate. Repeat until the reconstruction error stops decreasing. It's a
local optimizer — it can land in different local optima depending on the random init, which is
exactly why we run many seeds.

### Fit

```python
fit = 1 - ||X − X_reconstructed||_F / ||X||_F
```

A scalar in [0, 1]. Higher is better. 1.0 = exact reconstruction.

### Factor Match Score (FMS) — `tlviz.factor_tools.factor_match_score`

For two CP models from different random inits, FMS measures how *similar* the resulting
components are (with permutation/sign ambiguity handled internally). FMS = 1 means the two
runs found essentially the same decomposition. **Low FMS across runs at the same rank ⇒ the
decomposition is unstable at this rank**, often a sign that the rank is too high. High FMS ⇒
the model is well-identified and the components mean something.

### Choosing the "best" rank

Two signals, looked at together:
1. **Fit vs. R** — should rise then plateau. The elbow is the natural choice.
2. **FMS distribution at each R** — should stay high. If FMS collapses at R = 8 but is fine at
   R = 6, R = 6 is more trustworthy.

For DBLP with this setup, R between 4 and 10 is usually the interpretable sweet spot —
typically around R = 6.

---

## 7. Reconstructing the tensor and predicting 2005

```python
from tensorly.cp_tensor import cp_to_tensor
X_hat = cp_to_tensor((weights, factors))       # full reconstructed tensor
y_pred_score = X_hat[:, :, -3:].mean(axis=2)   # average last three years
```

**Why average the last three time-points?** It's a poor man's forecast. CP gives us a *smooth*
model of the temporal mode, so the reconstructed tensor is denoised. Averaging the most recent
years assumes 2005 looks like a continuation of 2002–2004 — a reasonable assumption for a slow
domain like academic publishing.

We then score predictions against the binarized `Y` with ROC/AUC, the same way we did for SVD.

---

## 8. What is "the best model"?

For this notebook we use CP on the full 3-way tensor. The rank-selection logic is:

| Signal | What we read |
|---|---|
| **Fit vs R** | Should rise then plateau. The elbow is the natural choice. |
| **Mean pairwise FMS at each R** | Should stay high. A collapse at some R means the decomposition is unstable at that rank — the algorithm is finding different local optima from different inits, so the components aren't reliably identifiable. |
| **2005-prediction AUC** | Treat as the downstream sanity check. The R with the best AUC is *usable* even if it's not the most stable, but I'd prefer an R where fit, FMS, and AUC are all reasonable rather than one that wins on AUC alone. |

For this DBLP tensor, the interpretable sweet spot is typically **R ≈ 6**: enough components
to capture distinct research communities, while pairwise FMS at that rank is usually still
high (≥0.9). At larger R, fit keeps creeping up but the components start splitting and
duplicating, and FMS collapses — a clear sign of overfitting.

**Final pick for understanding the data**: CP at the R where fit-elbow meets stable FMS
(typically R ≈ 6 for this dataset).
**Final pick for prediction**: the R from the sweep with the highest AUC on `Y` when scoring
by `mean(X_hat[:, :, -3:])`.

*(For context: SVD-on-flattened-Z is the standard baseline — it's covered in §§ 3–5 above as
background. The reason CP beats it in spirit is that CP keeps the year mode explicit, so you
can both inspect the temporal profile of each component and use it for forecasting. SVD has to
collapse the year mode before it even starts.)*

---

## 9. References

- Dunlavy, Kolda & Acar (2011) — *Temporal link prediction using matrix and tensor
  factorizations.* The paper this assignment is built on.
- Kolda & Bader (2009) — *Tensor decompositions and applications.* The canonical CP/Tucker
  reference.
- Kossaifi et al. (2019) — *TensorLy: Tensor Learning in Python.* The library docs.
- Roald & Moe (2022) — *TLViz.* The visualisation/FMS library.
