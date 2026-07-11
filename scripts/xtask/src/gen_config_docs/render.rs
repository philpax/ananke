//! Markdown rendering from the descriptor table.
//!
//! Produces `docs/configuration.md` by interleaving hand-written prose
//! fragments with field-reference tables generated from
//! `ananke_config::docs::all_sections()`.

use std::fmt::Write;

use ananke_config::docs::SectionDoc;

/// Prose fragments transcluded at compile time by `mod.rs`.
pub struct ProseFragments {
    pub overview: &'static str,
    pub daemon: &'static str,
    pub openai_api: &'static str,
    pub defaults: &'static str,
    pub devices: &'static str,
    pub service_common: &'static str,
    pub service_lifecycle: &'static str,
    pub service_placement: &'static str,
    pub health: &'static str,
    pub resource_allocation: &'static str,
    pub filters: &'static str,
    pub metadata: &'static str,
    pub tracking: &'static str,
    pub auto_restart: &'static str,
    pub auto_restart_error_rate: &'static str,
    pub auto_restart_periodic: &'static str,
    pub auto_restart_ttft_stall: &'static str,
    pub auto_restart_generation_stall: &'static str,
    pub auto_restart_footer: &'static str,
    pub llama_cpp: &'static str,
    pub estimation: &'static str,
    pub sampling: &'static str,
    pub command: &'static str,
    pub openai_proxy: &'static str,
    pub inheritance: &'static str,
}

/// Render the full `docs/configuration.md` document from the descriptor
/// table and prose fragments.
pub fn render_markdown(sections: &[SectionDoc], prose: &ProseFragments) -> String {
    let mut out = String::with_capacity(64 * 1024);

    render_overview(&mut out, prose);
    render_daemon(&mut out, sections, prose);
    render_openai_api(&mut out, sections, prose);
    render_defaults(&mut out, sections, prose);
    render_devices(&mut out, sections, prose);
    render_service_config(&mut out, sections, prose);
    render_health(&mut out, sections, prose);
    render_resource_allocation(&mut out, sections, prose);
    render_filters(&mut out, sections, prose);
    render_metadata(&mut out, prose);
    render_tracking(&mut out, sections, prose);
    render_auto_restart(&mut out, sections, prose);
    render_templates(&mut out, sections, prose);
    render_inheritance(&mut out, prose);

    // Collapse any triple+ newlines to double.
    while out.contains("\n\n\n") {
        out = out.replace("\n\n\n", "\n\n");
    }
    out
}

fn render_overview(out: &mut String, prose: &ProseFragments) {
    write_prose(out, prose.overview);
}

fn render_daemon(out: &mut String, sections: &[SectionDoc], prose: &ProseFragments) {
    emit_section(out, sections, "daemon", prose.daemon);
}

fn render_openai_api(out: &mut String, sections: &[SectionDoc], prose: &ProseFragments) {
    emit_section(out, sections, "openai_api", prose.openai_api);
}

fn render_defaults(out: &mut String, sections: &[SectionDoc], prose: &ProseFragments) {
    writeln!(out, "## Global Defaults").unwrap();
    write_prose(out, prose.defaults);
    emit_table(out, find_section(sections, "defaults"));
    writeln!(out).unwrap();
}

fn render_devices(out: &mut String, sections: &[SectionDoc], prose: &ProseFragments) {
    writeln!(out, "## Device Configuration").unwrap();
    write_prose(out, prose.devices);
    emit_table(out, find_section(sections, "devices"));
    writeln!(out).unwrap();
}

fn render_service_config(out: &mut String, sections: &[SectionDoc], prose: &ProseFragments) {
    writeln!(out, "## Service Configuration").unwrap();
    write_prose(out, prose.service_common);
    emit_table(out, find_section(sections, "service_common"));
    writeln!(out).unwrap();
    write_prose(out, prose.service_lifecycle);
    emit_table(out, find_section(sections, "service_devices"));
    writeln!(out).unwrap();
    write_prose(out, prose.service_placement);
}

fn render_health(out: &mut String, sections: &[SectionDoc], prose: &ProseFragments) {
    writeln!(out, "### Health Checks").unwrap();
    write_prose(out, prose.health);
    emit_table(out, find_section(sections, "service_health"));
    writeln!(out).unwrap();
}

fn render_resource_allocation(out: &mut String, sections: &[SectionDoc], prose: &ProseFragments) {
    writeln!(out, "### Resource Allocation").unwrap();
    write_prose(out, prose.resource_allocation);
    emit_table(out, find_section(sections, "service_allocation"));
    writeln!(out).unwrap();
}

fn render_filters(out: &mut String, sections: &[SectionDoc], prose: &ProseFragments) {
    emit_subsection(
        out,
        sections,
        "service_filters",
        "Request Filters",
        prose.filters,
    );
}

fn render_metadata(out: &mut String, prose: &ProseFragments) {
    writeln!(out, "### Metadata").unwrap();
    write_prose(out, prose.metadata);
}

fn render_tracking(out: &mut String, sections: &[SectionDoc], prose: &ProseFragments) {
    emit_subsection(
        out,
        sections,
        "service_tracking",
        "Tracking",
        prose.tracking,
    );
}

fn render_auto_restart(out: &mut String, sections: &[SectionDoc], prose: &ProseFragments) {
    emit_subsection(
        out,
        sections,
        "service_auto_restart",
        "Auto-restart",
        prose.auto_restart,
    );
    write_prose(out, prose.auto_restart_error_rate);
    emit_table(
        out,
        find_section(sections, "service_auto_restart_error_rate"),
    );
    writeln!(out).unwrap();
    write_prose(out, prose.auto_restart_periodic);
    emit_table(out, find_section(sections, "service_auto_restart_periodic"));
    writeln!(out).unwrap();
    write_prose(out, prose.auto_restart_ttft_stall);
    emit_table(
        out,
        find_section(sections, "service_auto_restart_ttft_stall"),
    );
    writeln!(out).unwrap();
    write_prose(out, prose.auto_restart_generation_stall);
    emit_table(
        out,
        find_section(sections, "service_auto_restart_generation_stall"),
    );
    writeln!(out).unwrap();
    write_prose(out, prose.auto_restart_footer);
}

fn render_templates(out: &mut String, sections: &[SectionDoc], prose: &ProseFragments) {
    writeln!(out, "## Templates").unwrap();
    render_llama_cpp(out, sections, prose);
    render_command(out, sections, prose);
}

fn render_llama_cpp(out: &mut String, sections: &[SectionDoc], prose: &ProseFragments) {
    writeln!(out, "### llama-cpp").unwrap();
    write_prose(out, prose.llama_cpp);
    // The llama_cpp prose ends with "#### Field Reference"; emit the
    // main field table here, then the sub-tables below.
    emit_table(out, find_section(sections, "llama_cpp"));
    writeln!(out).unwrap();
    // Estimation overrides.
    writeln!(out, "#### Estimation Overrides").unwrap();
    write_prose(out, prose.estimation);
    emit_table(out, find_section(sections, "llama_cpp_estimation"));
    writeln!(out).unwrap();
    // Sampling.
    writeln!(out, "#### Sampling").unwrap();
    write_prose(out, prose.sampling);
    emit_table(out, find_section(sections, "llama_cpp_sampling"));
    writeln!(out).unwrap();
}

fn render_command(out: &mut String, sections: &[SectionDoc], prose: &ProseFragments) {
    writeln!(out, "### command").unwrap();
    write_prose(out, prose.command);
    emit_table(out, find_section(sections, "command"));
    writeln!(out).unwrap();
    emit_subsection(
        out,
        sections,
        "openai_proxy",
        "OpenAI Proxy",
        prose.openai_proxy,
    );
}

fn render_inheritance(out: &mut String, prose: &ProseFragments) {
    writeln!(out, "## Service Inheritance").unwrap();
    write_prose(out, prose.inheritance);
}

// ── helpers ─────────────────────────────────────────────────────────────

/// Write a prose fragment (from `include_str!`), trimmed of trailing
/// whitespace, followed by a blank line separator.
fn write_prose(out: &mut String, s: &str) {
    writeln!(out, "{}\n", s.trim_end()).unwrap();
}

/// Emit a `##`-level section: heading, optional prose, then the table.
fn emit_section(out: &mut String, sections: &[SectionDoc], id: &str, prose: &str) {
    let section = find_section(sections, id);
    writeln!(out, "## {}\n", section.title).unwrap();
    if !prose.is_empty() {
        write_prose(out, prose);
    }
    emit_table(out, section);
    writeln!(out).unwrap();
}

/// Emit a `###`-level subsection: heading, optional prose, then the table.
fn emit_subsection(out: &mut String, sections: &[SectionDoc], id: &str, title: &str, prose: &str) {
    writeln!(out, "### {}\n", title).unwrap();
    if !prose.is_empty() {
        write_prose(out, prose);
    }
    emit_table(out, find_section(sections, id));
    writeln!(out).unwrap();
}

/// Find a section by id, panicking if missing (programming error in the
/// descriptor table).
fn find_section<'a>(sections: &'a [SectionDoc], id: &str) -> &'a SectionDoc {
    sections
        .iter()
        .find(|s| s.id == id)
        .unwrap_or_else(|| panic!("descriptor table missing section `{id}`"))
}

/// Render a field-reference table from a `SectionDoc`.
fn emit_table(out: &mut String, section: &SectionDoc) {
    writeln!(out, "| Field | Type | Default | Description |").unwrap();
    writeln!(out, "| --- | --- | --- | --- |").unwrap();
    for f in &section.fields {
        writeln!(
            out,
            "| `{}` | {} | {} | {} |",
            f.name, f.ty, f.default, f.description,
        )
        .unwrap();
    }
}
