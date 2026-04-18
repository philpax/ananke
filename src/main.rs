use ananke::daemon::run;
use ananke::errors::ExpectedError;

// Silence unused-import warning until `run` returns a real `ExpectedError`; keeps
// the type in scope so future signature changes surface here first.
#[allow(dead_code)]
fn _ensure_error_type_in_scope() -> Option<ExpectedError> { None }

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
