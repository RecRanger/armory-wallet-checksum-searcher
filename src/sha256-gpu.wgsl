// SHA-256 for 32-bit aligned messages.
// https://github.com/MarcoCiaramella/sha256-gpu/blob/main/index.js @ e8c7227

fn swap_endianess32(val: u32) -> u32 {
    return ((val>>24u) & 0xffu) | ((val>>8u) & 0xff00u) | ((val<<8u) & 0xff0000u) | ((val<<24u) & 0xff000000u);
}

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

// Inject `device.limits().max_compute_workgroup_size_x` value from Rust.
override CONFIG_WORKGROUP_SIZE: u32 = 256u;

// Set to true to enable 2-round sha256d. Set to false to keep as single-round sha256.
override CONFIG_ENABLE_SHA256D: bool = false;

@group(0) @binding(0) var<storage, read_write> messages: array<u32>;
@group(0) @binding(1) var<storage, read> num_messages: u32;

/// `message_sizes[0]` is the original message length in bytes.
/// `message_sizes[1]` is the padded message length in bytes.
@group(0) @binding(2) var<storage, read> message_sizes: array<u32>;

/// `hashes` contains the output hashes of each message, tightly packed.
@group(0) @binding(3) var<storage, read_write> hashes: array<u32>;



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

// ---------- Compression ----------

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


@compute @workgroup_size(CONFIG_WORKGROUP_SIZE)
fn sha256_or_sha256d(@builtin(global_invocation_id) gid: vec3<u32>) {

    let index = gid.x;
    if (index >= num_messages) { return; }

    let words_per_message = message_sizes[1] / 4u;
    let msg_base = index * words_per_message;
    let hash_base = index * 8u;

    // ===== FIRST ROUND =====

    var h = array<u32,8>(
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    );

    let num_chunks = (message_sizes[1] * 8u) / 512u;

    for (var c = 0u; c < num_chunks; c++) {
        var w = array<u32,64>();

        let chunk = msg_base + c * 16u;
        for (var i = 0u; i < 16u; i++) {
            w[i] = swap_endianess32(messages[chunk + i]);
        }
        for (var i = 16u; i < 64u; i++) {
            w[i] = w[i-16u] + g0(w[i-15u]) + w[i-7u] + g1(w[i-2u]);
        }

        sha256_compress(&w, &h);
    }

    // ===== OPTIONAL SECOND ROUND (compile-time) =====

    if (CONFIG_ENABLE_SHA256D) {
        var w2 = array<u32,64>();

        for (var i = 0u; i < 8u; i++) {
            w2[i] = h[i];
        }

        w2[8] = 0x80000000u;
        for (var i = 9u; i < 15u; i++) {
            w2[i] = 0u;
        }
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

    // ===== OUTPUT =====

    for (var i = 0u; i < 8u; i++) {
        hashes[hash_base + i] = swap_endianess32(h[i]);
    }
}