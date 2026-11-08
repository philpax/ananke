// Pure formatting helpers for rendering a LaunchCommand as a
// copy-pasteable shell command, used by LaunchCommandPanel.tsx.
// Containerized workloads render the runtime create command plus the
// in-container argv distinctly from native process argv.

import type { LaunchCommand } from "../../../api/client.ts";

export function renderCommand(cmd: LaunchCommand): string {
  if (cmd.container) {
    return renderContainerCommand(cmd);
  }
  return renderProcessCommand(cmd);
}

function renderProcessCommand(cmd: LaunchCommand): string {
  const lines: string[] = [];
  if (cmd.env.length > 0) {
    if (cmd.env_inherit) {
      lines.push("# inherits daemon env + overrides below");
    } else {
      lines.push("# clean env — only vars below");
    }
  }
  for (const e of cmd.env) lines.push(`${e.key}=${shellQuote(e.value)}`);
  const [binary, ...rest] = cmd.argv;
  if (binary !== undefined) lines.push(shellQuote(binary));
  for (let i = 0; i < rest.length; i++) {
    const tok = rest[i]!;
    const next = rest[i + 1];
    if (tok.startsWith("-") && next !== undefined && !next.startsWith("-")) {
      lines.push(`  ${shellQuote(tok)} ${shellQuote(next)}`);
      i++;
    } else {
      lines.push(`  ${shellQuote(tok)}`);
    }
  }
  return lines.join(" \\\n");
}

function renderContainerCommand(cmd: LaunchCommand): string {
  const c = cmd.container;
  if (!c) return renderProcessCommand(cmd);
  // The only runnable command here is the runtime `create`; the
  // in-container argv is its own tail, and is shown as a comment rather
  // than a second line. Emitting it as a command would mean pasting this
  // preview runs the workload on the host as well as creating the
  // container.
  const header = [
    `# ${c.runtime} create — the container is started separately`,
    `# in-container command: ${c.argv.map(shellQuote).join(" ")}`,
  ].join("\n");

  const [runtime, ...rest] = c.create_argv;
  const lines: string[] = [];
  if (runtime !== undefined) lines.push(shellQuote(runtime));
  for (let i = 0; i < rest.length; i++) {
    const tok = rest[i]!;
    const next = rest[i + 1];
    if (tok.startsWith("-") && next !== undefined && !next.startsWith("-")) {
      lines.push(`  ${shellQuote(tok)} ${shellQuote(next)}`);
      i++;
    } else {
      lines.push(`  ${shellQuote(tok)}`);
    }
  }
  return `${header}\n${lines.join(" \\\n")}`;
}

export function shellQuote(s: string): string {
  if (s.length === 0) return "''";
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(s)) return s;
  return `'${s.replace(/'/g, "'\\''")}'`;
}
