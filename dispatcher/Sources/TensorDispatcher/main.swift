import Foundation

enum ComputeEngine: String, Equatable {
    case cpuAMX
    case gpuMetal
    case npuANE
}

struct GraphProfile {
    let fusedDepth: Int
    let channels: Int
    let spatial: Int
    let requiresFP32: Bool

    /// One fp16 1×1 kernel, in mebibytes. Not the chain.
    var perLayerWeightMB: Double {
        Double(channels * channels * 2) / 1_048_576.0
    }

    var totalWeightMB: Double {
        perLayerWeightMB * Double(fusedDepth)
    }
}

enum TensorDispatcher {
    /// Routes from the shapes timed on this M4 Max.
    ///
    /// Short-spatial chains kept getting faster on the Neural Engine as total
    /// weights grew from 16 MB (32-deep, 7.0 TFLOPS) to 64 MB (128-deep, 12.9).
    /// A 24 MB total-weight ceiling would send that peak chain to the GPU,
    /// which ran it at 6.6 TFLOPS. The spill we measured was one 4096-channel
    /// kernel at spatial 64 (2.1 TFLOPS), not deep 512-channel chains.
    ///
    /// Wide spatial axes (2048 and the 4096³ GEMM) won on the fused GPU graph
    /// even at depth 1 and depth 4. Depth alone does not mark a GPU problem.
    static func route(_ profile: GraphProfile) -> ComputeEngine {
        if profile.requiresFP32 {
            return .cpuAMX
        }
        if profile.spatial >= 2048 {
            return .gpuMetal
        }
        if profile.fusedDepth == 1 && profile.channels <= 1024 {
            return .cpuAMX
        }
        if profile.fusedDepth >= 32 && profile.channels <= 512 && profile.spatial <= 64 {
            return .npuANE
        }
        if profile.fusedDepth == 1 && profile.spatial <= 64 && profile.channels >= 2048 {
            return .npuANE
        }
        return .cpuAMX
    }

    /// The blueprint as specified: no spatial axis, per-layer weight, depth-1
    /// always CPU, depth >= 32 and per-layer weight <= 24 MB to the Neural Engine.
    static func routeBlueprint(_ profile: GraphProfile) -> ComputeEngine {
        let perLayer = profile.perLayerWeightMB
        if profile.requiresFP32 || profile.fusedDepth == 1 {
            return .cpuAMX
        }
        if profile.fusedDepth >= 32 && perLayer <= 24.0 {
            return .npuANE
        }
        if profile.fusedDepth >= 16 {
            return .gpuMetal
        }
        return .cpuAMX
    }
}

struct MeasuredCase {
    let profile: GraphProfile
    let winner: ComputeEngine
    let note: String
}

let measured: [MeasuredCase] = [
    .init(profile: .init(fusedDepth: 1, channels: 512, spatial: 64, requiresFP32: false),
          winner: .cpuAMX, note: "0.023 ms CPU, 0.089 ms ANE"),
    .init(profile: .init(fusedDepth: 1, channels: 1024, spatial: 64, requiresFP32: false),
          winner: .cpuAMX, note: "0.079 ms CPU, 0.089 ms ANE"),
    .init(profile: .init(fusedDepth: 1, channels: 2048, spatial: 64, requiresFP32: false),
          winner: .npuANE, note: "0.174 ms ANE, 0.389 ms CPU"),
    .init(profile: .init(fusedDepth: 32, channels: 512, spatial: 64, requiresFP32: false),
          winner: .npuANE, note: "0.154 ms ANE, 0.92 ms CPU, GPU graph 0.19 TFLOPS"),
    .init(profile: .init(fusedDepth: 64, channels: 512, spatial: 64, requiresFP32: false),
          winner: .npuANE, note: "10.0 TFLOPS ANE, 5.8 GPU, 1.1 CPU"),
    .init(profile: .init(fusedDepth: 128, channels: 512, spatial: 64, requiresFP32: false),
          winner: .npuANE, note: "12.9 TFLOPS ANE at 64 MB total weights"),
    .init(profile: .init(fusedDepth: 256, channels: 256, spatial: 64, requiresFP32: false),
          winner: .npuANE, note: "9.7 TFLOPS ANE, 5.8 GPU, 1.2 CPU"),
    .init(profile: .init(fusedDepth: 16, channels: 512, spatial: 2048, requiresFP32: false),
          winner: .gpuMetal, note: "16.5 TFLOPS GPU, 3.2 CPU"),
    .init(profile: .init(fusedDepth: 4, channels: 2048, spatial: 2048, requiresFP32: false),
          winner: .gpuMetal, note: "14.6 TFLOPS GPU, 3.4 CPU"),
    .init(profile: .init(fusedDepth: 1, channels: 4096, spatial: 4096, requiresFP32: false),
          winner: .gpuMetal, note: "7.8 TFLOPS GPU, 3.0 CPU"),
    .init(profile: .init(fusedDepth: 128, channels: 512, spatial: 64, requiresFP32: true),
          winner: .cpuAMX, note: "fp32 was only timed on the CPU"),
]

var failures = 0
var blueprintMisses = 0
print("shape                    winner     routed     blueprint  weights")
for row in measured {
    let p = row.profile
    let got = TensorDispatcher.route(p)
    let blueprint = TensorDispatcher.routeBlueprint(p)
    let shape = "\(p.fusedDepth)x \(p.channels)ch sp\(p.spatial)\(p.requiresFP32 ? " fp32" : "")"
    let mark = got == row.winner ? "ok" : "MISS"
    if got != row.winner { failures += 1 }
    if blueprint != row.winner { blueprintMisses += 1 }
    let weights = String(format: "per-layer %.2f MB, chain %.1f MB", p.perLayerWeightMB, p.totalWeightMB)
    print("\(shape.padding(toLength: 24, withPad: " ", startingAt: 0)) \(row.winner.rawValue.padding(toLength: 10, withPad: " ", startingAt: 0)) \(got.rawValue.padding(toLength: 10, withPad: " ", startingAt: 0)) \(blueprint.rawValue.padding(toLength: 10, withPad: " ", startingAt: 0)) \(weights)  \(mark)  \(row.note)")
}
print("routed mismatches \(failures), blueprint mismatches \(blueprintMisses)")
if failures != 0 {
    exit(1)
}
