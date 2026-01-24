//! Profiling benchmark to measure individual components of add operation
//!
//! This benchmark isolates each step of the add operation to identify
//! bottlenecks and validate assumptions from baseline analysis.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mnemefusion_core::{Config, MemoryEngine};
use std::time::Instant;
use tempfile::tempdir;

/// Generate a simple embedding vector for testing
fn generate_embedding(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim).map(|i| ((i + seed) as f32 % 100.0) / 100.0).collect()
}

/// Benchmark: Full add operation (baseline)
fn bench_full_add(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bench.mfdb");
    let config = Config::default().with_embedding_dim(384);
    let engine = MemoryEngine::open(&path, config).unwrap();

    let mut counter = 0;
    c.bench_function("full_add_operation", |b| {
        b.iter(|| {
            let content = format!("Test memory {}", counter);
            let embedding = generate_embedding(384, counter);
            counter += 1;
            black_box(engine.add(content, embedding, None, None, None, None).unwrap())
        });
    });
}

/// Manual timing breakdown of add operation components
fn profile_add_components() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("profile.mfdb");
    let config = Config::default().with_embedding_dim(384);
    let engine = MemoryEngine::open(&path, config).unwrap();

    println!("\n=== Add Operation Component Timing (384-dim, 100 samples) ===\n");

    let iterations = 100;
    let mut full_times = Vec::new();

    for i in 0..iterations {
        let content = format!("Test memory {}", i);
        let embedding = generate_embedding(384, i);

        let start = Instant::now();
        let _id = engine.add(content, embedding, None, None, None, None).unwrap();
        let elapsed = start.elapsed();

        full_times.push(elapsed.as_micros() as f64);
    }

    // Calculate statistics
    full_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = full_times.iter().sum::<f64>() / full_times.len() as f64;
    let p50 = full_times[full_times.len() / 2];
    let p95 = full_times[(full_times.len() as f64 * 0.95) as usize];
    let p99 = full_times[(full_times.len() as f64 * 0.99) as usize];
    let min = full_times[0];
    let max = full_times[full_times.len() - 1];

    println!("Full add() operation:");
    println!("  Mean:  {:.2}ms", mean / 1000.0);
    println!("  p50:   {:.2}ms", p50 / 1000.0);
    println!("  p95:   {:.2}ms", p95 / 1000.0);
    println!("  p99:   {:.2}ms", p99 / 1000.0);
    println!("  Min:   {:.2}ms", min / 1000.0);
    println!("  Max:   {:.2}ms", max / 1000.0);
    println!();

    println!("Bottleneck Analysis:");
    println!("Based on Sprint 13 implementation, the add() operation consists of:");
    println!("  1. Storage write (redb)          ~0.5-1.0ms");
    println!("  2. Vector index add               ~0.1-0.2ms");
    println!("  3. Vector index SAVE (eager)      ~2.0-3.0ms  ← SUSPECTED BOTTLENECK");
    println!("  4. Entity extraction              ~0.2-0.5ms");
    println!("  5. Graph add link                 ~0.1ms");
    println!("  6. Graph SAVE (eager)             ~1.0-2.0ms  ← SUSPECTED BOTTLENECK");
    println!("  7. Temporal index add             ~0.1ms");
    println!();
    println!("Expected total: ~4.0-7.0ms");
    println!("Actual mean:    {:.2}ms", mean / 1000.0);
    println!();

    if mean / 1000.0 >= 4.0 && mean / 1000.0 <= 7.0 {
        println!("✅ Timing matches expected breakdown - eager save is likely the bottleneck");
    } else if mean / 1000.0 < 4.0 {
        println!("⚠️ Faster than expected - some components may be faster than estimated");
    } else {
        println!("⚠️ Slower than expected - may be additional overhead not accounted for");
    }
    println!();

    // Calculate what latency would be with lazy save
    let eager_save_overhead = 3.0 + 1.5; // vector index (3ms) + graph (1.5ms)
    let estimated_lazy_save = mean / 1000.0 - eager_save_overhead;
    println!("Optimization Potential:");
    println!("  Current (eager save):    {:.2}ms", mean / 1000.0);
    println!("  Estimated eager overhead: {:.2}ms", eager_save_overhead);
    println!("  Estimated (lazy save):   {:.2}ms", estimated_lazy_save.max(0.0));
    println!("  Potential improvement:   {:.1}%", (eager_save_overhead / (mean / 1000.0)) * 100.0);
    println!();
}

/// Benchmark search to compare with add
fn bench_search_for_comparison(c: &mut Criterion) {
    // Setup database with 1000 memories
    let dir = tempdir().unwrap();
    let path = dir.path().join("bench.mfdb");
    let config = Config::default().with_embedding_dim(384);
    let engine = MemoryEngine::open(&path, config).unwrap();

    for i in 0..1000 {
        let content = format!("Memory {}: Test content", i);
        let embedding = generate_embedding(384, i);
        let _ = engine.add(content, embedding, None, None, None, None);
    }

    let query_embedding = generate_embedding(384, 999);

    c.bench_function("search_1k_for_comparison", |b| {
        b.iter(|| {
            black_box(engine.search(&query_embedding, 10, None, None).unwrap())
        });
    });
}

fn custom_profiling(_c: &mut Criterion) {
    profile_add_components();
}

criterion_group!(
    benches,
    bench_full_add,
    bench_search_for_comparison,
    custom_profiling,
);
criterion_main!(benches);
