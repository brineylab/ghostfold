<agents>

  <overview>
    This document defines how an agent should operate within the GhostFold repository. Instructions may be incomplete, ambiguous, or conflicting. The agent is expected to resolve conflicts using the priority rules below and proceed without blocking on perfect clarity.
  </overview>

  <priorities>
    <rule order="1">Preserve correctness of scientific and computational results.</rule>
    <rule order="2">Avoid breaking existing public APIs, CLI behavior, and test expectations.</rule>
    <rule order="3">Minimize scope of changes; prefer local fixes over broad refactors.</rule>
    <rule order="4">Keep behavior consistent with existing patterns in the codebase.</rule>
    <rule order="5">Prefer clarity and maintainability over cleverness.</rule>
  </priorities>

  <fallbacks>
    <case name="ambiguous_instruction">
      Choose the interpretation that aligns with existing code patterns and tests.
    </case>

```
<case name="conflicting_instructions">
  Follow the higher-priority rule from the <priorities> section. If priorities do not resolve the conflict, preserve existing behavior.
</case>

<case name="missing_context">
  Infer intent from neighboring modules, tests, and naming conventions. Do not invent new abstractions unless absolutely necessary.
</case>

<case name="uncertain_change_scope">
  Implement the smallest change that satisfies the requirement. Avoid speculative improvements.
</case>

<case name="performance_vs_correctness">
  Default to correctness. Apply performance optimizations only if they do not change expected outputs.
</case>

<case name="test_failures">
  Treat failing tests as ground truth unless clearly outdated. Fix code before modifying tests.
</case>
```

  </fallbacks>

<project_structure>
GhostFold follows a src-layout Python package design.

```
<paths>
  <core>src/ghostfold/</core>
  <cli>src/ghostfold/cli/</cli>
  <pipeline>src/ghostfold/core/</pipeline>
  <msa>src/ghostfold/msa/</msa>
  <io>src/ghostfold/io/</io>
  <viz>src/ghostfold/viz/</viz>
  <mutator>src/ghostfold/mutator/</mutator>
  <tests>tests/</tests>
  <config>src/ghostfold/data/default_config.yaml</config>
  <scripts>scripts/</scripts>
</paths>

Tests mirror module structure where possible.
```

</project_structure>

  <commands>
    <install>pip install -e ".[dev]"</install>
    <test_all>pytest</test_all>
    <test_single>pytest tests/test_cli.py -q</test_single>
    <lint>ruff check src tests</lint>
    <cli_help>ghostfold --help</cli_help>
    <cli_module>python -m ghostfold.cli.app --help</cli_module>
    <build>python -m build</build>
  </commands>

  <architecture>
    <purpose>
      Generate synthetic MSAs from single sequences using ProstT5, then perform structure prediction with local ColabFold.
    </purpose>

```
<pipeline_flow>
  CLI → run_pipeline → msa generation → filtering → mutation (optional) → visualization → ColabFold execution
</pipeline_flow>

<modules>
  <module name="cli/app.py">CLI entrypoints (Typer)</module>
  <module name="core/pipeline.py">Orchestration and caching</module>
  <module name="msa/generation.py">Batched sequence generation</module>
  <module name="msa/model.py">Model loading</module>
  <module name="msa/filters.py">Filtering and deduplication</module>
  <module name="msa/neff.py">Neff computation</module>
  <module name="core/colabfold.py">ColabFold subprocess wrapper</module>
  <module name="core/config.py">Config loading and merging</module>
</modules>
```

  </architecture>

  <api>
    <policy>
      Public API stability is expected. Avoid renaming or removing exported functions without strong justification.
    </policy>

```
<functions>
  run_pipeline, read_fasta_from_path, collect_fasta_paths, mask_a3m_file, calculate_neff, MSA_Mutator
</functions>
```

  </api>

  <style>
    <rules>
      Use 4-space indentation.
      Use snake_case for functions and variables.
      Use PascalCase for classes.
      Keep modules small and focused.
      Add type hints where they clarify intent.
    </rules>

    <separation>
      CLI logic stays in src/ghostfold/cli/. Core logic must not depend on CLI modules.
    </separation>
  </style>

  <testing>
    <framework>pytest</framework>

```
<policy>
  Tests define expected behavior. Do not change tests to match incorrect implementations unless clearly justified.
</policy>

<guidelines>
  Name files as test_<feature>.py.
  Name tests as test_<behavior>.
  Prefer fast unit tests.
  Use typer.testing.CliRunner for CLI validation.
</guidelines>
```

  </testing>

  <contributions>
    <commits>
      Use short, imperative messages.
      Keep commits focused on a single concern.
    </commits>

```
<pull_requests>
  Include summary, relevant context, test evidence, and CLI examples when applicable.
</pull_requests>
```

  </contributions>

  <performance>
    <principle>
      Performance improvements must not change observable outputs unless explicitly intended.
    </principle>

```
<msa_generation>
  Coverage levels are batched into a single generation call per decoding configuration.
</msa_generation>

<deduplication>
  Uses NumPy-based Hamming identity; falls back for unequal lengths.
</deduplication>

<neff>
  Pairwise identity computed in blocks to limit memory usage.
</neff>

<precision>
  Uses bfloat16 or float16 depending on hardware.
</precision>

<oom>
  Batch size is reduced and retried on failure.
</oom>
```

  </performance>

<runtime_notes> <requirements>
Local ColabFold installation with CUDA-enabled PyTorch is required. </requirements>

```
<logging>
  Rich is used for logging and progress display.
</logging>

<config>
  inference_batch_size controls memory behavior.
</config>

<security>
  Do not hardcode machine-specific paths, credentials, or tokens.
</security>
```

</runtime_notes>

</agents>
