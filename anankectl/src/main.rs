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
        default_value = ananke_api::defaults::MANAGEMENT_ENDPOINT
    )]
    endpoint: String,

    /// Emit responses as raw JSON instead of formatted text.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// Show the current configuration.
    Show,
    /// Validate a configuration file or stdin.
    Validate {
        /// Path to the configuration file (reads stdin if not provided).
        file: Option<std::path::PathBuf>,
    },
    /// Reload the configuration.
    Reload,
}

#[derive(Subcommand)]
enum OneshotCommand {
    /// Submit a oneshot job from a TOML file.
    Submit {
        /// Path to the TOML file describing the job.
        file: std::path::PathBuf,
    },
    /// Submit a oneshot job built from inline flags and a trailing command.
    Run {
        /// Optional human-readable name.
        #[arg(long)]
        name: Option<String>,
        /// Eviction priority (higher wins).
        #[arg(long, default_value_t = 50)]
        priority: u8,
        /// Time-to-live duration string (e.g. "2h", "30m").
        #[arg(long)]
        ttl: Option<String>,
        /// Working directory for the spawned child.
        #[arg(long)]
        workdir: Option<std::path::PathBuf>,
        /// Device-placement mode.
        #[arg(long, default_value = "gpu-only")]
        placement: String,
        /// Static VRAM allocation in GiB; conflicts with --min-vram-gb/--max-vram-gb.
        #[arg(long, conflicts_with_all = ["min_vram_gb", "max_vram_gb"])]
        vram_gb: Option<f32>,
        /// Dynamic lower bound for VRAM in GiB; requires --max-vram-gb.
        #[arg(long, requires = "max_vram_gb")]
        min_vram_gb: Option<f32>,
        /// Dynamic upper bound for VRAM in GiB.
        #[arg(long)]
        max_vram_gb: Option<f32>,
        /// Command and arguments to run.
        #[arg(trailing_var_arg = true, required = true)]
        command: Vec<String>,
    },
    /// List all known oneshot jobs.
    List,
    /// Cancel a oneshot job by ID.
    Kill {
        /// Oneshot job ID to cancel.
        id: String,
    },
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
    /// Start a service.
    Start {
        /// Service name.
        name: String,
    },
    /// Stop a service.
    Stop {
        /// Service name.
        name: String,
    },
    /// Restart a service.
    Restart {
        /// Service name.
        name: String,
    },
    /// Enable a service.
    Enable {
        /// Service name.
        name: String,
    },
    /// Disable a service.
    Disable {
        /// Service name.
        name: String,
    },
    /// Retry a service (enable then start).
    Retry {
        /// Service name.
        name: String,
    },
    /// Tail logs for a service.
    Logs {
        /// Service name.
        name: String,
        /// Follow new lines as they arrive.
        #[arg(long)]
        follow: bool,
        /// Filter to a specific run id.
        #[arg(long)]
        run: Option<i64>,
        /// Minimum timestamp (ms since epoch).
        #[arg(long)]
        since: Option<i64>,
        /// Maximum timestamp (ms since epoch).
        #[arg(long)]
        until: Option<i64>,
        /// Cap on number of historical lines returned.
        #[arg(long, default_value_t = 200)]
        limit: u32,
        /// Filter to stdout or stderr.
        #[arg(long)]
        stream: Option<String>,
    },
    /// Manage oneshot jobs.
    Oneshot {
        #[command(subcommand)]
        command: OneshotCommand,
    },
    /// Manage configuration.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Reload configuration (alias for `config reload`).
    Reload,
    /// Talk to a model via the OpenAI-compatible API.
    Chat {
        /// Model (service) name.
        model: String,
        /// User prompt.
        #[arg(trailing_var_arg = true)]
        prompt: Vec<String>,
        /// System prompt.
        #[arg(long, default_value = "You are a helpful assistant.")]
        system_prompt: String,
    },
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let client = client::ApiClient::new(&cli.endpoint);
    let result = match cli.command {
        Command::Devices => commands::devices::run(&client, cli.json).await,
        Command::Services { all } => commands::services::run(&client, cli.json, all).await,
        Command::Show { name } => commands::show::run(&client, cli.json, &name).await,
        Command::Start { name } => commands::lifecycle::start(&client, cli.json, &name).await,
        Command::Stop { name } => commands::lifecycle::stop(&client, cli.json, &name).await,
        Command::Restart { name } => commands::lifecycle::restart(&client, cli.json, &name).await,
        Command::Enable { name } => commands::lifecycle::enable(&client, cli.json, &name).await,
        Command::Disable { name } => commands::lifecycle::disable(&client, cli.json, &name).await,
        Command::Retry { name } => commands::lifecycle::retry(&client, cli.json, &name).await,
        Command::Logs {
            name,
            follow,
            run,
            since,
            until,
            limit,
            stream,
        } => {
            commands::logs::run(
                &client, cli.json, &name, follow, run, since, until, limit, stream,
            )
            .await
        }
        Command::Oneshot { command } => match command {
            OneshotCommand::Submit { file } => {
                commands::oneshot::submit(&client, cli.json, &file).await
            }
            OneshotCommand::Run {
                name,
                priority,
                ttl,
                workdir,
                placement,
                vram_gb,
                min_vram_gb,
                max_vram_gb,
                command,
            } => {
                commands::oneshot::run(
                    &client,
                    cli.json,
                    name,
                    priority,
                    ttl,
                    workdir,
                    placement,
                    vram_gb,
                    min_vram_gb,
                    max_vram_gb,
                    command,
                )
                .await
            }
            OneshotCommand::List => commands::oneshot::list(&client, cli.json).await,
            OneshotCommand::Kill { id } => commands::oneshot::kill(&client, cli.json, &id).await,
        },
        Command::Config { command } => match command {
            ConfigCommand::Show => commands::config::show(&client, cli.json).await,
            ConfigCommand::Validate { file } => {
                commands::config::validate(&client, cli.json, file.as_deref()).await
            }
            ConfigCommand::Reload => commands::config::reload(&client, cli.json).await,
        },
        Command::Reload => commands::config::reload(&client, cli.json).await,
        Command::Chat {
            model,
            prompt,
            system_prompt,
        } => {
            let prompt = prompt.join(" ");
            commands::chat::run(&client, cli.json, &model, &prompt, &system_prompt).await
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("anankectl: {e}");
            e.exit_code()
        }
    }
}
