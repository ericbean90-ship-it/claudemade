// swift-tools-version: 5.9
// Swift Package Manager 설정 — DialogWatcher 빌드 정의
import PackageDescription

let package = Package(
    name: "DialogWatcher",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "DialogWatcher",
            path: "Sources/DialogWatcher"
        )
    ]
)
