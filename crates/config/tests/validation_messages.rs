//! Every config validation message, pinned.
//!
//! Validation builds each message with `format!` at the rule that rejects the
//! config, which keeps the rule and its wording together but leaves the wording
//! itself unasserted: the existing tests check that a config is rejected, and
//! at most that the message contains some substring. A message can lose the
//! list of values it exists to enumerate, or trade its rationale for a syntax
//! hint, without any test noticing.
//!
//! Update `validation_messages.txt` when a message legitimately changes, and
//! read the diff as the operator will. Re-pin with
//! `UPDATE_EXPECT=1 cargo test -p ananke-config --test validation_messages`.

use std::path::Path;

use ananke_config::load_config_from_str;

const CORPUS: &[&str] = &[
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\n",
    "[[service]]\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nport = 1\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\nnuma = \"x\"\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\nexpert_offload = \"x\"\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\ndevices.split = \"x\"\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\nlifecycle = \"x\"\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\nmodality = \"x\"\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\ndraft_model = \"/d.gguf\"\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\nlauncher = []\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\nauto_restart = { spec_collapse = true }\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\nauto_restart = { periodic = true }\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\nidle_timeout = \"zzz\"\n",
    "[[service]]\nname = \"a\"\ntemplate = \"command\"\ncommand = []\nport = 1\nallocation.mode = \"static\"\nallocation.reserve_gb = 1\n",
    "[[service]]\nname = \"a\"\ntemplate = \"command\"\ncommand = [\"x\"]\nport = 1\n",
    "[[service]]\nname = \"a\"\ntemplate = \"command\"\ncommand = [\"x\"]\nport = 1\nallocation.mode = \"dynamic\"\n",
    "[[service]]\nname = \"a\"\ntemplate = \"command\"\ncommand = [\"x\"]\nport = 1\nallocation = { mode = \"dynamic\", min_reserve_gb = 9, max_reserve_gb = 2 }\n",
    "[[service]]\nname = \"a\"\ntemplate = \"command\"\ncommand = [\"x\"]\nport = 1\nallocation.mode = \"zzz\"\n",
    "[[service]]\nname = \"a\"\ntemplate = \"command\"\ncommand = [\"x\"]\nport = 1\nallocation.mode = \"static\"\nallocation.reserve_gb = 1\ntracking.cgroup_parent = \"rel\"\n",
    "[[service]]\nname = \"a\"\ntemplate = \"command\"\ncommand = [\"x\"]\nport = 1\nallocation.mode = \"static\"\nallocation.reserve_gb = 1\ntracking.cgroup_parent = \"/a b!\"\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\n[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/y.gguf\"\nport = 2\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\n[[service]]\nname = \"b\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/y.gguf\"\nport = 1\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\nextends = \"nope\"\n",
    "[daemon]\nprivate_port_start = 60000\nprivate_port_end = 50000\n",
    "[daemon]\nmanagement_listen = \"0.0.0.0:7071\"\n",
    "[daemon]\nshutdown_timeout = \"zzz\"\n",
    "[daemon]\nmanagement_listen = \"nope\"\n",
    "this is not toml [[[",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\ndevices.placement = \"cpu-only\"\nn_gpu_layers = 40\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\ndevices.placement = \"zzz\"\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\ndevices.placement_override = {}\n",
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\ndevices.split = \"tensor\"\ndevices.placement = \"hybrid\"\n",
    "[[service]]\nname = \"reservation:1\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\n",
];
/// Path of the pinned rendering, relative to this test file.
const EXPECTED: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/validation_messages.txt");

/// `index|message` for the error each input produces. Validation stops at the
/// first failure, so each rejected input contributes exactly one line.
fn render() -> String {
    let mut out = String::new();
    for (i, src) in CORPUS.iter().enumerate() {
        if let Err(error) = load_config_from_str(src, Path::new("/config.toml")) {
            out.push_str(&format!(
                "{i:03}|{}\n",
                error.to_string().replace('\n', "\\n")
            ));
        }
    }
    out
}

#[test]
fn validation_messages_match_the_pinned_rendering() {
    let actual = render();
    if std::env::var_os("UPDATE_EXPECT").is_some() {
        std::fs::write(EXPECTED, &actual).expect("write pinned rendering");
        return;
    }
    let expected = std::fs::read_to_string(EXPECTED).expect("read pinned rendering");
    if actual != expected {
        for (a, e) in actual.lines().zip(expected.lines()) {
            if a != e {
                eprintln!("actual:   {a}\nexpected: {e}");
            }
        }
        panic!(
            "rendered validation messages changed; review the diff, then re-pin with \
             UPDATE_EXPECT=1 cargo test -p ananke-config --test validation_messages"
        );
    }
}
