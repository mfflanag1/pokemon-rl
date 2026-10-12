# Bounded statistics implementation

Own only python/pokemon_switch_screen/statistics.py. You are not alone in the
codebase; do not revert or change other files. No delegation. NumPy and stdlib only.
Implement summarize(candidate, baseline, *, alpha=.05, family_size=6).
Inputs numeric integer arrays [N,2] with 2<=N<=100000; exact values -1=cutoff,
0=loss,1=win. Reject bool/object/complex/float dtypes, mismatched shapes, invalid
values, nonfinite or malformed parameters with ValueError. alpha finite real
0<alpha<1 (not bool); family_size builtin/NumPy integer 1..10000 (not bool).
Return JSON-native finite values, exact keys:
pairs,games,wins,losses,cutoffs,outcome_lower,outcome_upper,baseline_lower,
baseline_upper,secured_win_change,worst_case_improvement,best_case_improvement,
adjusted_win_lower,adjusted_improvement_lower.
Use means of complete two-seat pairs. outcome lower=mean(candidate==1), upper=
mean(candidate!=0); analogous baseline. Worst improvement=mean(candwin-base!=0),
best=mean(cand!=0-basewin). secured change=mean(candwin-basewin).
radius=sqrt((log(family_size)-log(alpha))/(2*N)), with alpha/family underflow avoided.
adjusted_win_lower=max(0,outcome_lower-radius);
adjusted_improvement_lower=max(-1,worst_case_improvement-2*radius).
These are one-sided Hoeffding bounds for ranges1 and2; no bootstrap, no p-values,
no counts/CI among terminal-only rows and never recode cutoff as draw.
Frozen tests are provided. Add no files. Do not modify tests or plan.
