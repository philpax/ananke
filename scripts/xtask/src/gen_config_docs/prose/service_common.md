Services are defined as an array of `[[service]]` blocks. Each service uses one of two templates: `llama-cpp` (for GGUF models via llama.cpp) or `command` (for arbitrary binaries or Docker wrappers).

### Common Fields

These fields appear at the top level of every `[[service]]` block, regardless of template:
