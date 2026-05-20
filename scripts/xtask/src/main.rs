//! Workspace automation for ananke.
//!
//! Invoked via the `xtask` cargo alias from the workspace root, e.g.
//! `cargo xtask release 0.2.0`. Keep new subcommands narrow and prefer
//! shell-outs to git/cargo over re-implementing their behaviour.

use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod release;

#[derive(Parser)]
#[command(name = "xtask", about = "ananke workspace automation")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Bump the workspace version, commit the change, and create a v-tag locally.
    Release(release::Args),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Release(args) => release::run(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(1)
        }
    }
}
