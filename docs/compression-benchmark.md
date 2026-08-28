# NetCDF Compression Benchmark

Benchmark of the first `calc_TEA` calculation from the SPARTACUS workflow
(`Tx30`, 2021-2026, `significant_digits: 3`, `compression_level: 1`). The
compressed cases used `zlib_compression: true`; the uncompressed cases used
`zlib_compression: false`.

## Write Benchmark

| Location | zlib | Runtime | Logical size | Allocated size |
|---|---:|---:|---:|---:|
| `/data` | On | 2:37 | 415M | 215M |
| `/data` | Off | 4:38 | 19G | 2.2G |
| `/tmp` | On | 2:37 | 434M | 415M |
| `/tmp` | Off | 2:05 | 19.7G | 19G |

The `/data` sizes were measured on the ZFS server. The ZFS dataset containing
the data reported `101T` used, `125T` logical used, and a dataset compression
ratio of `1.28x`. The `/tmp` files were not on that ZFS dataset.

## Read Benchmark

Three alternating trials were run. Values below are medians for reading both
the daily basis-variable and CTP NetCDF files with `xarray.Dataset.load()`.

| Location | zlib | Daily load | CTP load | Total read |
|---|---:|---:|---:|---:|
| `/data` | On | 25.0 s | 0.32 s | 27.9 s |
| `/data` | Off | 11.4 s | 0.06 s | 14.1 s |
| `/tmp` | On | 24.8 s | 0.34 s | 28.2 s |
| `/tmp` | Off | 11.5 s | 0.06 s | 13.3 s |

Peak RSS during the daily-file reads was approximately 18-19 GiB. The
compressed files took roughly twice as long to read, but used approximately
45 times less logical space. The `/data` and `/tmp` timings were broadly
similar.

The read timings were influenced by filesystem and page caches; no system
page-cache flush was performed. The write and read measurements were separate
runs and should be treated as representative single-run benchmarks rather
than controlled cold-cache measurements.
