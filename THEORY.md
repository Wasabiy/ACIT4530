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
author–conference links appear next year. The framing follows Dunlavy, Kolda & Acar (2011)
[D-K-A 2011].

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

**Why this is the "best" rank-K approximation.** The Eckart–Young theorem [Eckart-Young 1936;
Mirsky 1960]: among all matrices of rank ≤ K, `Z_k` minimizes both the Frobenius and spectral
reconstruction error. SVD gives us this optimum in closed form [Golub-VanLoan 2013, §2.4].

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
every possible threshold [Fawcett 2006].

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

**What CP does.** It factors a 3-way tensor `X` into a sum of `R` rank-one tensors
[Hitchcock 1927; Carroll-Chang 1970; Harshman 1970; K&B 2009]:

```
X[i,j,k] ≈ Σ_{r=1..R}  λ_r · A[i,r] · B[j,r] · C[k,r]
```

So each component `r` is a triple of vectors — one per mode (authors, conferences, years) —
plus a weight `λ_r`. Reading a component means looking at all three vectors together: the
authors that load high on `A[:,r]`, the conferences that load high on `B[:,r]`, and the
temporal shape `C[:,r]`. That's a *theme* — a group of authors who publish at a group of
conferences with a particular time profile.

**Why CP and not Tucker?** Tucker decomposition [Tucker 1966] uses a core tensor and is more
flexible but **non-unique**. CP is essentially unique under mild conditions (Kruskal's theorem)
[Kruskal 1977; Sidiropoulos-Bro 2000], which is what makes its components *interpretable*.
That uniqueness is the whole reason CP is the workhorse for exploratory tensor analysis
[K&B 2009, §3.2].

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
third by least squares; rotate. Repeat until the reconstruction error stops decreasing
[Carroll-Chang 1970; Harshman 1970; K&B 2009, §3.4]. It's a local optimizer — it can land in
different local optima depending on the random init, which is exactly why we run many seeds.

### Fit

```python
fit = 1 - ||X − X_reconstructed||_F / ||X||_F
```

A scalar in [0, 1]. Higher is better. 1.0 = exact reconstruction.

### Factor Match Score (FMS) — `tlviz.factor_tools.factor_match_score`

For two CP models from different random inits, FMS measures how *similar* the resulting
components are (with permutation/sign ambiguity handled internally) [Bro-Kiers 2003;
Roald-Moe 2022]. FMS = 1 means the two runs found essentially the same decomposition.
**Low FMS across runs at the same rank ⇒ the decomposition is unstable at this rank**, often
a sign that the rank is too high. High FMS ⇒ the model is well-identified and the components
mean something.

### Choosing the "best" rank

Two signals, looked at together:
1. **Fit vs. R** — should rise then plateau. The elbow is the natural choice.
2. **FMS distribution at each R** — should stay high. If FMS collapses at R = 8 but is fine at
   R = 6, R = 6 is more trustworthy.

For DBLP with this setup, R between 4 and 10 is usually the interpretable sweet spot —
typically around R = 6.

---

## 6b. Nonnegative CP vs unconstrained CP

Plain `parafac` places **no constraints** on the factor entries — `A`, `B`, `C` can be any
real numbers, positive or negative. `tensorly.decomposition.non_negative_parafac` adds the
constraint `A, B, C ≥ 0` everywhere [Lee-Seung 1999; Lee-Seung 2001; Welling-Weber 2001;
Cichocki et al. 2009; K&B 2009, §4.3].

### Why the unconstrained version produces negative loadings

CP has a built-in **sign ambiguity**: within any one component you can flip the sign of any
two of the three factor vectors and the rank-one term `λ_r · a_r ⊗ b_r ⊗ c_r` is unchanged
[K&B 2009, §3.2]. On top of that, individual entries inside a single factor vector are also
unconstrained, so a component can legitimately have a positive loading on some conferences
and a negative loading on others. In the DBLP fit at R = 6 you can see this in Figure 6:
`CONF/IPPS` sits with a small *negative* loading next to a wall of positive EDA venues. The
model is saying "authors active on this component publish at DAC/ICCAD/etc. **instead of**
IPPS" — a contrast pattern, not a positivity pattern.

### Why that contrast helps fit

Unconstrained CP has a richer hypothesis space. A single component can encode a
"do X, don't do Y" contrast (positive on X, negative on Y), which lets the model explain
more variance per component. Nonnegative CP cannot represent contrasts inside one component
— "X but not Y" has to be split across two components, or it gets pushed into the residual.
For the same rank R, unconstrained CP therefore typically attains a strictly better
reconstruction error [K&B 2009, §4.3; Cichocki et al. 2009].

### Why the nonnegative version is often more interpretable

When `A, B, C ≥ 0`, each component becomes a strictly additive part: "how much" of an
author/conference/year, never "instead of." This is the same intuition as nonnegative matrix
factorisation [Lee-Seung 1999; Lee-Seung 2001] — components behave like soft cluster
memberships rather than signed contrasts. There is no sign-flip ambiguity, the scale is
fixed by the nonnegativity, and the components compose by summation, which is what most
people mean when they say "an interpretable decomposition." Nonnegative CP is also identifiable
under milder conditions than unconstrained CP in some settings [Lim-Comon 2009].

### The actual trade-off

| Axis | Unconstrained CP | Nonnegative CP |
|---|---|---|
| Fit at fixed R | Equal or better — strictly larger feasible set | Equal or worse |
| Interpretability | Loadings are "directional" — signs encode contrast | Loadings are "amounts" — strictly additive parts |
| Sign / scale identifiability | Sign ambiguity within each component | Fully resolved — no sign flips, scale fixed by `≥ 0` |
| Algorithm | ALS with unconstrained least-squares subproblems | ALS with nonnegative least squares (NNLS) or multiplicative updates [Lee-Seung 2001; Cichocki et al. 2009] |
| Typical # iterations | Lower | Higher — NNLS is more expensive per ALS sweep |
| Use case in this assignment | Best for the **R-sweep / fit / FMS / AUC** experiments — we want the cleanest measurement of how rank affects fit | Best for the **report figure** at R = 6 if we want loadings that read as "how much" without having to explain negative bars |

### How this connects to Figure 6 in the notebook

`top_k` sorts by `np.abs(loading)`, so IPPS shows up because `|−0.05|` is larger than the
next positive value, not because IPPS is "more important than DAC." Sorting by magnitude is
the standard CP-reading convention precisely because of the sign ambiguity. If we wanted the
figure to be a pure "top-10 most loaded conferences, no negatives," the cleanest fix is to
re-fit with `non_negative_parafac` at the same R; the EDA story does not change but the
display is unambiguous.

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
domain like academic publishing [D-K-A 2011, §3].

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

## 8b. Do we need to save the fitted CP models?

Short answer: **no, caching is a convenience, not a correctness requirement.** The
ranking of R values — i.e. which R "wins" on each criterion — is robust to the random
init across the seed range we use.

Three reasons the ordering is stable:

1. **Fit is essentially seed-invariant at fixed R.** ALS is a local optimizer, but on
   this 471 × 366 × 14 tensor the loss landscape has a dominant basin per rank: every
   seed lands at the same final fit to four decimal places (e.g. R=2 fit = 0.0429 ±
   0.0000 across 15 seeds; in the R-sweep, `mean_fit` and `best_fit` differ by ≤0.002).
   So when we plot "fit vs R" the curve is the same whatever seeds we ran.

2. **Fit is monotone in R.** More components can always explain more variance, so
   the fit ordering across R is **R=1 < R=4 < R=6 < … < R=50** every time. No seed
   ever flips this. The R that maximises fit is always R=50 — seed-independent.

3. **The interpretation rank (R≈6) is chosen by a *robust* criterion, not the
   argmax.** We pick R where fit-elbow meets stable FMS — both are stable summaries
   over multiple seeds, not a single noisy run. The elbow region (R=4–10) doesn't
   move between seed batches.

**Where seeds *do* matter — and what we do about it.**

The one place seed noise can flip the answer is the **best-R-for-AUC** in §7. The
AUC vs R curve is fairly flat across R=6–15, so on any single run the *argmax* might
land at R=8 one time and R=10 the next — differences often <0.01 AUC, well within
seed noise. The fix is not to cache the exact model, but to:

- Report the AUC *region* ("AUC peaks in R=6–15 around 0.X, degraded at R=50"), not
  the single argmax R.
- Or bump `N_INITS_SWEEP` so each R's representative is averaged over more seeds —
  this stabilises the argmax at near-zero extra report-writing cost.

**So why offer the cache at all?** Pure ergonomics. Re-fitting the full sweep takes
~2–3 minutes; loading from `joblib` is ~1 second. The cache lets you iterate on
plots and discussion without re-paying that cost, and it also pins the *exact* fits
across TensorLy/NumPy version changes for full numerical reproducibility. But
nothing in the report's conclusions depends on which seeds you happen to have run —
the science is in the shapes of the fit/FMS/AUC curves, and those are seed-stable.

---

## 9. References

In-text citations use short tags (e.g. `[K&B 2009]`); the full entries are below, grouped by
topic.

### The assignment paper

- **[D-K-A 2011]** Dunlavy, D. M., Kolda, T. G., & Acar, E. (2011). *Temporal link prediction
  using matrix and tensor factorizations.* ACM Transactions on Knowledge Discovery from Data,
  5(2), Article 10. doi:10.1145/1921632.1921636. — Frames the DBLP author × conference × year
  prediction task and the "average last-N reconstructed slices" forecasting recipe used in § 7.

### CP / PARAFAC core references

- **[K&B 2009]** Kolda, T. G., & Bader, B. W. (2009). *Tensor decompositions and applications.*
  SIAM Review, 51(3), 455–500. doi:10.1137/07070111X. — The canonical survey: CP, Tucker, ALS,
  uniqueness, nonnegative variants.
- **[Hitchcock 1927]** Hitchcock, F. L. (1927). *The expression of a tensor or a polyadic as
  a sum of products.* Journal of Mathematics and Physics, 6(1–4), 164–189. — The original CP
  idea ("polyadic decomposition").
- **[Carroll-Chang 1970]** Carroll, J. D., & Chang, J. J. (1970). *Analysis of individual
  differences in multidimensional scaling via an N-way generalization of "Eckart–Young"
  decomposition.* Psychometrika, 35(3), 283–319. — One of the two independent re-discoveries
  ("CANDECOMP") that established ALS for CP.
- **[Harshman 1970]** Harshman, R. A. (1970). *Foundations of the PARAFAC procedure: Models
  and conditions for an "explanatory" multi-modal factor analysis.* UCLA Working Papers in
  Phonetics, 16, 1–84. — The other independent re-discovery ("PARAFAC").
- **[Tucker 1966]** Tucker, L. R. (1966). *Some mathematical notes on three-mode factor
  analysis.* Psychometrika, 31(3), 279–311. — Defines the Tucker decomposition (the
  non-unique, more flexible cousin of CP).

### Uniqueness and identifiability

- **[Kruskal 1977]** Kruskal, J. B. (1977). *Three-way arrays: Rank and uniqueness of
  trilinear decompositions, with application to arithmetic complexity and statistics.* Linear
  Algebra and its Applications, 18(2), 95–138. — Kruskal's sufficient condition for CP
  uniqueness.
- **[Sidiropoulos-Bro 2000]** Sidiropoulos, N. D., & Bro, R. (2000). *On the uniqueness of
  multilinear decomposition of N-way arrays.* Journal of Chemometrics, 14(3), 229–239. —
  Generalises Kruskal's condition to N-way tensors.

### Nonnegative factorisation (§ 6b)

- **[Lee-Seung 1999]** Lee, D. D., & Seung, H. S. (1999). *Learning the parts of objects by
  non-negative matrix factorization.* Nature, 401(6755), 788–791. — Introduces NMF and the
  "parts-based representation" intuition that carries over to nonnegative CP.
- **[Lee-Seung 2001]** Lee, D. D., & Seung, H. S. (2001). *Algorithms for non-negative matrix
  factorization.* Advances in Neural Information Processing Systems (NeurIPS) 13, 556–562. —
  Multiplicative-update algorithms used inside many nonnegative-CP implementations.
- **[Welling-Weber 2001]** Welling, M., & Weber, M. (2001). *Positive tensor factorization.*
  Pattern Recognition Letters, 22(12), 1255–1261. — Early extension of NMF to 3-way tensors.
- **[Cichocki et al. 2009]** Cichocki, A., Zdunek, R., Phan, A. H., & Amari, S. (2009).
  *Nonnegative matrix and tensor factorizations: Applications to exploratory multi-way data
  analysis and blind source separation.* Wiley. — Book-length treatment of nonnegative
  matrix/tensor factorisation algorithms.
- **[Lim-Comon 2009]** Lim, L.-H., & Comon, P. (2009). *Nonnegative approximations of
  nonnegative tensors.* Journal of Chemometrics, 23(7–8), 432–441. — Identifiability and
  existence results specific to nonnegative tensor decomposition.

### SVD background (§§ 3–5)

- **[Eckart-Young 1936]** Eckart, C., & Young, G. (1936). *The approximation of one matrix by
  another of lower rank.* Psychometrika, 1(3), 211–218. — The optimality theorem for
  truncated SVD under the Frobenius norm.
- **[Mirsky 1960]** Mirsky, L. (1960). *Symmetric gauge functions and unitarily invariant
  norms.* The Quarterly Journal of Mathematics, 11(1), 50–59. — Extends Eckart–Young to all
  unitarily invariant norms (including the spectral norm).
- **[Golub-VanLoan 2013]** Golub, G. H., & Van Loan, C. F. (2013). *Matrix computations*
  (4th ed.). Johns Hopkins University Press. — Standard reference for the numerical SVD
  algorithm and the thin-SVD form used in § 4.

### Evaluation (§ 5)

- **[Fawcett 2006]** Fawcett, T. (2006). *An introduction to ROC analysis.* Pattern
  Recognition Letters, 27(8), 861–874. — Practical guide to ROC curves and AUC, including why
  AUC is preferred over accuracy on imbalanced data.

### Stability diagnostics (§ 6, FMS)

- **[Bro-Kiers 2003]** Bro, R., & Kiers, H. A. L. (2003). *A new efficient method for
  determining the number of components in PARAFAC models.* Journal of Chemometrics, 17(5),
  274–286. — The CORCONDIA diagnostic and the broader practice of measuring component
  similarity across runs that FMS formalises.

### Libraries used in the notebook

- **[Kossaifi et al. 2019]** Kossaifi, J., Panagakis, Y., Anandkumar, A., & Pantic, M. (2019).
  *TensorLy: Tensor Learning in Python.* Journal of Machine Learning Research, 20(26), 1–6. —
  The `tensorly` library, which provides `parafac` and `non_negative_parafac`.
- **[Roald-Moe 2022]** Roald, M., & Moe, Y. M. (2022). *TLViz: Visualising and analysing
  tensor decomposition models with Python.* Journal of Open Source Software, 7(78), 4754.
  doi:10.21105/joss.04754. — The `tlviz` library; provides `factor_match_score` used for
  FMS in § 6.
