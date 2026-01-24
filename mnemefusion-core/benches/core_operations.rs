//! Core operation benchmarks for MnemeFusion
//!
//! Benchmarks the most critical operations:
//! - Add memory (with vector indexing)
//! - Search (semantic similarity)
//! - Query (multi-dimensional fusion)
//! - Get by ID
//! - Delete

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mnemefusion_core::{Config, MemoryEngine};
use tempfile::tempdir;

/// Generate a simple embedding vector for testing
fn generate_embedding(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim).map(|i| ((i + seed) as f32 % 100.0) / 100.0).collect()
}

/// Create a test database with N memories
fn setup_database(n: usize, dim: usize) -> (MemoryEngine, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bench.mfdb");

    let config = Config::default().with_embedding_dim(dim);
    let engine = MemoryEngine::open(&path, config).unwrap();

    // Pre-populate with memories
    for i in 0..n {
        let content = format!("Memory {}: This is test content for benchmarking", i);
        let embedding = generate_embedding(dim, i);
        let _ = engine.add(content, embedding, None, None, None, None);
    }

    (engine, dir)
}

/// Benchmark: Add a single memory
fn bench_add_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_memory");

    for dim in [128, 384, 768].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("dim", dim), dim, |b, &dim| {
            let dir = tempdir().unwrap();
            let path = dir.path().join("bench.mfdb");
            let config = Config::default().with_embedding_dim(dim);
            let engine = MemoryEngine::open(&path, config).unwrap();

            let mut counter = 0;
            b.iter(|| {
                let content = format!("Test memory {}", counter);
                let embedding = generate_embedding(dim, counter);
                counter += 1;
                black_box(engine.add(content, embedding, None, None, None, None).unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark: Search operation at different scales
fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");

    for size in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("memories", size), size, |b, &size| {
            let (engine, _dir) = setup_database(size, 384);
            let query_embedding = generate_embedding(384, 999);

            b.iter(|| {
                black_box(engine.search(&query_embedding, 10, None, None).unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark: Query with fusion (multi-dimensional)
fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");

    for size in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("memories", size), size, |b, &size| {
            let (engine, _dir) = setup_database(size, 384);
            let query_embedding = generate_embedding(384, 999);

            b.iter(|| {
                black_box(
                    engine.query(
                        "What happened yesterday?",
                        &query_embedding,
                        10,
                        None,
                        None,
                    ).unwrap()
                )
            });
        });
    }

    group.finish();
}

/// Benchmark: Get by ID
fn bench_get_by_id(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_by_id");

    for size in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("memories", size), size, |b, &size| {
            let (engine, _dir) = setup_database(size, 384);

            // Get a random ID from the middle
            let test_content = format!("Memory {}: This is test content for benchmarking", size / 2);
            let test_embedding = generate_embedding(384, size / 2);
            let id = engine.add(test_content, test_embedding, None, None, None, None).unwrap();

            b.iter(|| {
                black_box(engine.get(&id).unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark: Delete operation
fn bench_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("delete");

    group.bench_function("delete_single", |b| {
        b.iter_batched(
            || {
                // Setup: create database with one memory
                let dir = tempdir().unwrap();
                let path = dir.path().join("bench.mfdb");
                let config = Config::default().with_embedding_dim(384);
                let engine = MemoryEngine::open(&path, config).unwrap();
                let embedding = generate_embedding(384, 0);
                let id = engine.add("Test memory".to_string(), embedding, None, None, None, None).unwrap();
                (engine, id, dir)
            },
            |(engine, id, _dir)| {
                // Benchmark: delete the memory
                black_box(engine.delete(&id, None).unwrap())
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmark: Batch add operation
fn bench_batch_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_add");

    for batch_size in [10, 100, 1_000].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(BenchmarkId::new("batch_size", batch_size), batch_size, |b, &batch_size| {
            b.iter_batched(
                || {
                    // Setup: create database and batch inputs
                    let dir = tempdir().unwrap();
                    let path = dir.path().join("bench.mfdb");
                    let config = Config::default().with_embedding_dim(384);
                    let engine = MemoryEngine::open(&path, config).unwrap();

                    let inputs: Vec<_> = (0..batch_size)
                        .map(|i| {
                            mnemefusion_core::types::MemoryInput::new(
                                format!("Batch memory {}", i),
                                generate_embedding(384, i),
                            )
                        })
                        .collect();

                    (engine, inputs, dir)
                },
                |(engine, inputs, _dir)| {
                    // Benchmark: add batch
                    black_box(engine.add_batch(inputs, None).unwrap())
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark: Temporal range query
fn bench_temporal_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_range");

    for size in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("memories", size), size, |b, &size| {
            let (engine, _dir) = setup_database(size, 384);

            // Query last 10% of memories by time
            let now = mnemefusion_core::types::Timestamp::now();
            let start = now.subtract_days(1);

            b.iter(|| {
                black_box(engine.get_range(start, now, 100, None).unwrap())
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_add_memory,
    bench_search,
    bench_query,
    bench_get_by_id,
    bench_delete,
    bench_batch_add,
    bench_temporal_range,
);

criterion_main!(benches);
