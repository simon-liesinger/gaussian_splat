import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var modelURL: URL? = {
        // Auto-open .ply from command line argument
        let args = CommandLine.arguments
        if args.count > 1 {
            let path = args[1]
            let url = URL(fileURLWithPath: path)
            if FileManager.default.fileExists(atPath: path) {
                return url
            }
        }
        return nil
    }()
    @State private var isPickingFile = false
    @State private var gaussianCount: Int = 0

    var body: some View {
        ZStack {
            if let url = modelURL {
                SplatView(url: url, gaussianCount: $gaussianCount)
                    .ignoresSafeArea()

                VStack {
                    HStack {
                        Spacer()
                        VStack(alignment: .trailing, spacing: 4) {
                            Text(url.lastPathComponent)
                                .font(.caption)
                                .foregroundColor(.white)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(.black.opacity(0.6))
                                .cornerRadius(6)
                            if gaussianCount > 0 {
                                Text("\(gaussianCount.formatted()) splats")
                                    .font(.caption2)
                                    .foregroundColor(.white.opacity(0.8))
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 2)
                                    .background(.black.opacity(0.4))
                                    .cornerRadius(4)
                            }
                        }
                        .padding(12)
                    }
                    Spacer()
                    HStack {
                        Text("Drag: orbit  |  Scroll: zoom  |  Right-drag: pan")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.5))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(.black.opacity(0.3))
                            .cornerRadius(6)
                    }
                    .padding(.bottom, 8)
                }
            } else {
                VStack(spacing: 20) {
                    Image(systemName: "cube.transparent")
                        .font(.system(size: 64))
                        .foregroundColor(.secondary)

                    Text("Gaussian Splat Viewer")
                        .font(.title)

                    Text("Open a .ply, .splat, or .spz file")
                        .foregroundColor(.secondary)

                    Button("Open File...") {
                        isPickingFile = true
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                }
            }
        }
        .fileImporter(
            isPresented: $isPickingFile,
            allowedContentTypes: [
                UTType(filenameExtension: "ply")!,
                UTType(filenameExtension: "splat")!,
                UTType(filenameExtension: "spz")!,
            ]
        ) { result in
            if case .success(let url) = result {
                _ = url.startAccessingSecurityScopedResource()
                modelURL = url
            }
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                if let url {
                    DispatchQueue.main.async {
                        self.modelURL = url
                    }
                }
            }
            return true
        }
        .frame(minWidth: 600, minHeight: 400)
    }
}
