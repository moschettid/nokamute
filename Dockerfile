FROM rust:1.87 AS builder

WORKDIR /app
COPY . .

RUN cargo build --release
CMD ["./target/release/nokamute"]
