import SwiftUI
import AppKit

@main
struct GaussianViewerApp: App {
    init() {
        // Force the process to be a foreground GUI app (needed for SPM executables)
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .defaultSize(width: 1200, height: 800)
    }
}
