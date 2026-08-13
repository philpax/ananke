//! Every diagnostic message the config pipeline can render, pinned.
//!
//! A rule and the sentence it produces are written at the same site, so a
//! message can be reworded — or quietly lost — without any typed assertion
//! noticing. Three messages degraded that way before this existed: two dropped
//! the list of values they were meant to enumerate, and one traded its
//! rationale for a syntax hint.
//!
//! Update `diagnostic_messages.txt` when a message legitimately changes, and
//! read the diff as the operator will.

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
    "[[service]]\nname = \"a\"\ntemplate = \"llama-cpp\"\nmodel = \"/m/x.gguf\"\nport = 1\nlifecycle = \"oneshot\"\n",
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
];
/// `index|code|path|fields|message` for every diagnostic each input produces.
fn render() -> String {
    let mut out = String::new();
    for (i, src) in CORPUS.iter().enumerate() {
        if let Err(report) = load_config_from_str(src) {
            for d in report.as_slice() {
                out.push_str(&format!(
                    "{i:03}|{code}|{path:?}|{fields:?}|{message}\n",
                    code = d.code(),
                    path = d.path(),
                    fields = d.fields(),
                    message = d.to_string().replace('\n', "\\n")
                ));
            }
        }
    }
    out
}

/// Path of the pinned rendering, relative to this test file.
const EXPECTED: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/diagnostic_messages.txt");

#[test]
fn diagnostic_messages_match_the_pinned_rendering() {
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
            "rendered diagnostics changed; review the diff, then re-pin with \
             UPDATE_EXPECT=1 cargo test -p ananke-config --test diagnostic_messages"
        );
    }
}
