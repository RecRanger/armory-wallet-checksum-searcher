// ============================================================
// Configuration
// ============================================================

// Inject from Rust
override CONFIG_WORKGROUP_SIZE: u32 = 256u;
override CONFIG_ENABLE_SHA256D: bool = true;
override CONFIG_STRIDE_X: u32; // threads per row

// ============================================================
// Bindings
// ============================================================

struct SearchConfig {
    input_len_bytes: u32, // Like 1_000_000.
    message_len_bytes: u32, // Like 20.
    compare_len_bytes: u32 // Like 4.
};

/// `input_bytes` should be in Big Endian words (tightly packed).
@group(0) @binding(0)
var<storage, read> input_bytes: array<u32>;

@group(0) @binding(1)
var<uniform> search_config: SearchConfig;

@group(0) @binding(2)
var<storage, read_write> match_offsets: array<u32>;

@group(0) @binding(3)
var<storage, read_write> match_count: atomic<u32>;

// ============================================================
// SHA-256 constants & helpers
// ============================================================

const K: array<u32, 64> = array<u32,64>(
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
);

fn shw(x: u32, n: u32) -> u32 {
    return (x << (n & 31u)) & 0xffffffffu;
}

fn r(x: u32, n: u32) -> u32 {
    return (x >> n) | shw(x, 32u - n);
}

fn g0(x: u32) -> u32 {
    return r(x, 7u) ^ r(x, 18u) ^ (x >> 3u);
}

fn g1(x: u32) -> u32 {
    return r(x, 17u) ^ r(x, 19u) ^ (x >> 10u);
}

fn s0(x: u32) -> u32 {
    return r(x, 2u) ^ r(x, 13u) ^ r(x, 22u);
}

fn s1(x: u32) -> u32 {
    return r(x, 6u) ^ r(x, 11u) ^ r(x, 25u);
}

fn maj(a: u32, b: u32, c: u32) -> u32 {
    return (a & b) ^ (a & c) ^ (b & c);
}

fn ch(e: u32, f: u32, g: u32) -> u32 {
    return (e & f) ^ ((~e) & g);
}

fn swap_endianess32(v: u32) -> u32 {
    return ((v >> 24u) & 0xffu)
         | ((v >> 8u)  & 0xff00u)
         | ((v << 8u)  & 0xff0000u)
         | ((v << 24u) & 0xff000000u);
}

// ============================================================
// Input helpers
// ============================================================

fn load_byte(base: u32, index: u32) -> u32 {
    let word = input_bytes[(base + index) >> 2u];
    let shift = (3u - ((base + index) & 3u)) * 8u;
    return (word >> shift) & 0xffu;
}

// ============================================================
// Padding helpers
// ============================================================

fn padded_len_bytes(message_len: u32) -> u32 {
    // Usage: select(falseValue, trueValue, condition)
    let total = message_len + 1u + 8u;
    let rem = total & 63u;
    let pad = select(64u - rem, 0u, rem == 0u);
    return total + pad;
}

// ============================================================
// Block construction (multi-block)
// ============================================================

fn build_block(
    offset: u32,
    block_index: u32,
    message_len: u32,
    w: ptr<function, array<u32,64>>
) {
    let total_len = padded_len_bytes(message_len);
    let block_base = block_index * 64u;

    for (var i = 0u; i < 64u; i++) {
        (*w)[i] = 0u;
    }

    // Pack message into `w` (block).
    for (var i = 0u; i < 64u; i++) {
        // Fill the first 16 words with the 64 bytes as available (or padding or length u64).
        let global_byte = block_base + i;
        var b = 0u;

        if (global_byte < message_len) {
            b = load_byte(offset, global_byte);
        } else if (global_byte == message_len) {
            b = 0x80u;
        } else if (global_byte >= total_len - 8u) {
            // Append the message length (in bits) in the last 8 bytes as a u64.
            // High 32 bits of the u64 are all zero.
            // Assume that the message length is in message_len fully as a u32.
            let bit_len = message_len * 8u;

            let byte_index = global_byte - (total_len - 8u);
            let shift = (7u - byte_index) * 8u;

            // Only the lowest 4 bytes can be non-zero under our assumption.
            if (shift < 32u) {
                b = (bit_len >> shift) & 0xffu;
            } else {
                // High 32 bits of the u64 are all zero.
                // TODO: Optimization: Could remove this assignment altogether.
                b = 0u;
            }
        }

        let word_index = i >> 2u; // Same as i/4.
        let shift = (3u - (i & 3u)) * 8u;
        (*w)[word_index] |= b << shift;
    }

    for (var i = 16u; i < 64u; i++) {
        w[i] = w[i-16u] + g0(w[i-15u]) + w[i-7u] + g1(w[i-2u]);
    }
}

// ============================================================
// Compression
// ============================================================

fn sha256_compress(w: ptr<function, array<u32,64>>,
                   h: ptr<function, array<u32,8>>) {
    var a = (*h)[0];
    var b = (*h)[1];
    var c = (*h)[2];
    var d = (*h)[3];
    var e = (*h)[4];
    var f = (*h)[5];
    var g = (*h)[6];
    var h0 = (*h)[7];

    for (var i = 0u; i < 64u; i++) {
        let t1 = h0 + s1(e) + ch(e,f,g) + K[i] + (*w)[i];
        let t2 = s0(a) + maj(a,b,c);
        h0 = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    (*h)[0] += a;
    (*h)[1] += b;
    (*h)[2] += c;
    (*h)[3] += d;
    (*h)[4] += e;
    (*h)[5] += f;
    (*h)[6] += g;
    (*h)[7] += h0;
}

// ============================================================
// Compute entry
// ============================================================

@compute @workgroup_size(CONFIG_WORKGROUP_SIZE, 1, 1)
fn sliding_sha256d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let offset = gid.y * CONFIG_STRIDE_X + gid.x;

    if (
        (offset + search_config.message_len_bytes + search_config.compare_len_bytes)
        > search_config.input_len_bytes
    ) {
        return;
    }

    var h = array<u32,8>(
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    );

    let blocks = padded_len_bytes(search_config.message_len_bytes) / 64u;

    for (var b = 0u; b < blocks; b++) {
        var w = array<u32,64>();
        build_block(offset, b, search_config.message_len_bytes, &w);
        sha256_compress(&w, &h);
    }

    if (CONFIG_ENABLE_SHA256D) {
        var w2 = array<u32,64>();
        for (var i = 0u; i < 8u; i++) { w2[i] = h[i]; }
        w2[8] = 0x80000000u;
        for (var i = 9u; i < 64u; i++) { w2[i] = 0u; }
        w2[15] = 256u;

        for (var i = 16u; i < 64u; i++) {
            w2[i] = w2[i-16u] + g0(w2[i-15u]) + w2[i-7u] + g1(w2[i-2u]);
        }

        h = array<u32,8>(
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
        );

        sha256_compress(&w2, &h);
    }

    // Compare prefix of hash against input bytes (little-endian).
    var mismatch = false;
    for (var i = 0u; i < search_config.compare_len_bytes; i++) {
        // Input byte: immediately after the message.
        let input_b = load_byte(offset, search_config.message_len_bytes + i);

        // Hash byte: little-endian byte order.
        let hash_word = swap_endianess32(h[i >> 2u]);
        let hash_shift = (i & 3u) * 8u;
        let hash_b = (hash_word >> hash_shift) & 0xffu;

        if (input_b != hash_b) {
            mismatch = true;
            break;
        }
    }
    if (!mismatch) {
        // This part (match is found) occurs very rarely (so it's fine to use slow atomic operations).
        let idx = atomicAdd(&match_count, 1u);
        if (idx < arrayLength(&match_offsets)) {
            match_offsets[idx] = offset;
        }
    }

    // DEBUG: Write out the hash.
    // if (offset == 275000) {
    //     for (var i = 0u; i < 32u; i += 4u) {
    //         let w = h[i / 4u]; // BE word value
    //         // let w = input_bytes[(offset + i) >> 2u];

    //         // Write LE bytes
    //         match_offsets[i + 3u] = (w >> 0u)  & 0xffu;
    //         match_offsets[i + 2u] = (w >> 8u)  & 0xffu;
    //         match_offsets[i + 1u] = (w >> 16u) & 0xffu;
    //         match_offsets[i + 0u] = (w >> 24u) & 0xffu;
    //     }
    // }
}
