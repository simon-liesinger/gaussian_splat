// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "GaussianViewer",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/scier/MetalSplatter.git", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "GaussianViewer",
            dependencies: [
                .product(name: "MetalSplatter", package: "MetalSplatter"),
                .product(name: "SplatIO", package: "MetalSplatter"),
                .product(name: "PLYIO", package: "MetalSplatter"),
            ]
        ),
    ]
)
