// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "CoreMLSmoke",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        .library(name: "CoreMLSmokeSupport", targets: ["CoreMLSmokeSupport"])
    ],
    targets: [
        .target(
            name: "CoreMLSmokeSupport"
        ),
        .testTarget(
            name: "CoreMLSmokeTests",
            dependencies: ["CoreMLSmokeSupport"],
            resources: [
                .copy("Resources")
            ]
        ),
    ]
)
