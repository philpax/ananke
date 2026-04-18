use ananke::daemon::run;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> std::process::ExitCode {
    match run().await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("ananke: {err}");
            std::process::ExitCode::from(err.exit_code())
        }
    }
}
