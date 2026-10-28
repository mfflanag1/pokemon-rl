"""F29: width-32 action-local nonlinear policy residual on the sealed F28 matchup host.

`network` holds the parameter inventory (`SHAPES`, `NEW_SHAPES`), the function-preserving
`extend`, `check_state` and the arm-dispatched `forward`; `learning` holds the unchanged
frozen-target PPO/Adam rules with a static `arm` in {'control', 'nonlinear'}.
"""
