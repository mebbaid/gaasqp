
# GaasQP

A geometric active-set QP solver for polytope-constrained quadratic programs. GaasQP exploits the polyhedral structure of the feasible region to find the optimal face via coordinate-antipodal vertex initialization and face-hopping, bypassing the need for a phase-I feasibility pass on warm-started problems.

Companion code for the paper *"Antipodal Initialization for Box-Constrained Quadratic Programs"*.

## Quick start

This project uses the `pixi` tool for dependency management and convenient run scripts (optional).

Installation (pixi):

```bash
# Install pixi (optional)
curl -fsSL https://pixi.sh/install.sh | bash
```

Run benchmarks:

```bash
# Reproduce paper randomized box QPs benchmark
pixi run benchmark-box > box_results.txt
pixi run python benchmarks/plot_benchmarks.py box_results.txt

# Reproduce paper DCM MPC benchmark
pixi run benchmark-lipm > lipm_results.txt
pixi run python benchmarks/plot_lipm_benchmark.py lipm_results.txt

```



## License

BSD 3-Clause License — see [LICENSE](LICENSE).

---


## Maintainers

<table align="left">
    <tr>
        <td><a href="https://github.com/mebbaid"><img src="https://github.com/mebbaid.png" width="40"></a></td>
        <td><a href="https://github.com/mebbaid">👨‍💻 @Mohamed Elobaid</a></td>
    </tr>
</table>

