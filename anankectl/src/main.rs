use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod client;
mod commands;
mod output;

#[derive(Parser)]
#[command(name = "anankectl", version)]
struct Cli {
    /// Base URL for the management API.
    #[arg(
        long,
        global = true,
        env = "ANANKE_ENDPOINT",
        default_value = "http://127.0.0.1:17777"
    )]
    endpoint: String,

    /// Emit responses as raw JSON instead of formatted text.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List devices with reservations.
    Devices,
    /// List services.
    Services {
        /// Include disabled services.
        #[arg(long)]
        all: bool,
    },
    /// Show service detail.
    Show {
        /// Service name.
        name: String,
    },
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let client = client::ApiClient::new(&cli.endpoint);
    let result = match cli.command {
        Command::Devices => commands::devices::run(&client, cli.json).await,
        Command::Services { all } => commands::services::run(&client, cli.json, all).await,
        Command::Show { name } => commands::show::run(&client, cli.json, &name).await,
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("anankectl: {e}");
            e.exit_code()
        }
    }
}
