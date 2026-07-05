```toml
[daemon]
management_listen = "0.0.0.0:7071"
allow_external_management = true # Required if management_listen is non-loopback
allow_external_services = true   # Allow public access to individual model ports
data_dir = "./data"
shutdown_timeout = "120s"        # Max time to wait for services to drain
private_port_start = 40000      # Start of loopback port range for private listeners
private_port_end = 59999        # End of loopback port range
llama_server = "/opt/llama-build/llama-server" # Default binary for every llama-cpp service
```

> **Security Note:** Both the Management API (`management_listen`) and per-service reverse proxies (`allow_external_services`) are **unauthenticated**. If you bind them to non-loopback addresses:
>
> - Trust your network perimeter (e.g., Tailscale, a private VLAN).
> - Terminate TLS and authentication at a reverse proxy in front of ananke.
> - Never expose these ports directly to the public internet.
