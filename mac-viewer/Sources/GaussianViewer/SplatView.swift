import SwiftUI
import MetalKit
import MetalSplatter
import SplatIO
import simd

/// Custom MTKView that forwards scroll events to the renderer
class ScrollableMTKView: MTKView {
    weak var scrollDelegate: SplatRenderer_?

    override func scrollWheel(with event: NSEvent) {
        scrollDelegate?.handleScroll(event)
    }

    override var acceptsFirstResponder: Bool { true }
}

struct SplatView: NSViewRepresentable {
    let url: URL
    @Binding var gaussianCount: Int

    func makeCoordinator() -> SplatRenderer_ {
        SplatRenderer_()
    }

    func makeNSView(context: Context) -> ScrollableMTKView {
        let view = ScrollableMTKView()
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device")
        }
        view.device = device
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = 1
        view.clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1)

        let renderer = context.coordinator
        renderer.setup(view: view)
        view.delegate = renderer
        view.scrollDelegate = renderer

        Task {
            let count = await renderer.loadModel(url: url, device: device, view: view)
            await MainActor.run {
                gaussianCount = count
            }
        }

        return view
    }

    func updateNSView(_ view: ScrollableMTKView, context: Context) {}
}

// MARK: - Orbit Camera Controller

@MainActor
class SplatRenderer_: NSObject, MTKViewDelegate {
    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var splatRenderer: MetalSplatter.SplatRenderer?
    private var drawableSize: CGSize = .zero

    // Orbit camera state
    private var yaw: Float = 0
    private var pitch: Float = -0.3
    private var distance: Float = 8
    private var panX: Float = 0
    private var panY: Float = 0

    // Mouse tracking
    private var lastMouseLocation: NSPoint = .zero
    private var isDragging = false
    private var isRightDragging = false

    private let inFlightSemaphore = DispatchSemaphore(value: 3)

    func setup(view: MTKView) {
        self.device = view.device!
        self.commandQueue = device.makeCommandQueue()!

        // Add gesture/event monitors
        let panGesture = NSPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        view.addGestureRecognizer(panGesture)

        // Scroll for zoom
        view.addTrackingArea(NSTrackingArea(
            rect: .zero,
            options: [.activeInKeyWindow, .inVisibleRect, .mouseMoved],
            owner: self
        ))

        // Set up scroll wheel monitoring on the view
        NotificationCenter.default.addObserver(
            forName: NSView.boundsDidChangeNotification,
            object: view,
            queue: .main
        ) { _ in }
    }

    func loadModel(url: URL, device: MTLDevice, view: MTKView) async -> Int {
        do {
            let renderer = try MetalSplatter.SplatRenderer(
                device: device,
                colorFormat: view.colorPixelFormat,
                depthFormat: view.depthStencilPixelFormat,
                sampleCount: view.sampleCount,
                maxViewCount: 1,
                maxSimultaneousRenders: 3
            )

            let reader = try AutodetectSceneReader(url)
            let points = try await reader.readAll()
            let chunk = try SplatChunk(device: device, from: points)
            await renderer.addChunk(chunk)

            self.splatRenderer = renderer
            return points.count
        } catch {
            print("Error loading model: \(error)")
            return 0
        }
    }

    // MARK: - Mouse handling

    @objc func handlePan(_ gesture: NSPanGestureRecognizer) {
        let translation = gesture.translation(in: gesture.view)
        let modifiers = NSEvent.modifierFlags

        if modifiers.contains(.control) || gesture.buttonMask == 0x2 {
            // Right-drag or ctrl-drag: pan
            panX += Float(translation.x) * 0.01
            panY -= Float(translation.y) * 0.01
        } else {
            // Left-drag: orbit
            yaw += Float(translation.x) * 0.005
            pitch += Float(translation.y) * 0.005
            pitch = max(-.pi / 2 + 0.01, min(.pi / 2 - 0.01, pitch))
        }

        gesture.setTranslation(.zero, in: gesture.view)
    }

    func handleScroll(_ event: NSEvent) {
        distance -= Float(event.scrollingDeltaY) * 0.1
        distance = max(0.5, min(50, distance))
    }

    // MARK: - Rendering

    private var viewport: MetalSplatter.SplatRenderer.ViewportDescriptor {
        let aspect = Float(drawableSize.width / drawableSize.height)
        let fovy: Float = 65.0 * .pi / 180.0

        let projectionMatrix = perspectiveMatrix(fovy: fovy, aspect: aspect, near: 0.1, far: 100.0)

        // Orbit camera: rotate around target, then translate
        let rotY = rotationY(yaw)
        let rotX = rotationX(pitch)
        let trans = translationMatrix(panX, panY, -distance)

        // Common calibration: flip Z and Y to match common 3DGS PLY orientation
        let calibration = rotationZ(.pi)

        let viewMatrix = trans * rotX * rotY * calibration

        let mtlViewport = MTLViewport(
            originX: 0, originY: 0,
            width: drawableSize.width, height: drawableSize.height,
            znear: 0, zfar: 1
        )

        return MetalSplatter.SplatRenderer.ViewportDescriptor(
            viewport: mtlViewport,
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix,
            screenSize: SIMD2(x: Int(drawableSize.width), y: Int(drawableSize.height))
        )
    }

    func draw(in view: MTKView) {
        guard let splatRenderer, splatRenderer.isReadyToRender else { return }
        guard let drawable = view.currentDrawable else { return }

        _ = inFlightSemaphore.wait(timeout: .distantFuture)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            return
        }

        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { _ in
            semaphore.signal()
        }

        do {
            let didRender = try splatRenderer.render(
                viewports: [viewport],
                colorTexture: view.multisampleColorTexture ?? drawable.texture,
                colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                depthTexture: view.depthStencilTexture,
                rasterizationRateMap: nil,
                renderTargetArrayLength: 0,
                to: commandBuffer
            )
            if didRender {
                commandBuffer.present(drawable)
            }
        } catch {
            print("Render error: \(error)")
        }

        commandBuffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
    }

    // MARK: - Matrix helpers

    func perspectiveMatrix(fovy: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let ys = 1 / tanf(fovy * 0.5)
        let xs = ys / aspect
        let zs = far / (near - far)
        return simd_float4x4(columns: (
            SIMD4(xs, 0,  0,  0),
            SIMD4(0,  ys, 0,  0),
            SIMD4(0,  0,  zs, -1),
            SIMD4(0,  0,  zs * near, 0)
        ))
    }

    func rotationX(_ angle: Float) -> simd_float4x4 {
        let c = cos(angle), s = sin(angle)
        return simd_float4x4(columns: (
            SIMD4(1, 0,  0, 0),
            SIMD4(0, c,  s, 0),
            SIMD4(0, -s, c, 0),
            SIMD4(0, 0,  0, 1)
        ))
    }

    func rotationY(_ angle: Float) -> simd_float4x4 {
        let c = cos(angle), s = sin(angle)
        return simd_float4x4(columns: (
            SIMD4(c, 0, -s, 0),
            SIMD4(0, 1,  0, 0),
            SIMD4(s, 0,  c, 0),
            SIMD4(0, 0,  0, 1)
        ))
    }

    func rotationZ(_ angle: Float) -> simd_float4x4 {
        let c = cos(angle), s = sin(angle)
        return simd_float4x4(columns: (
            SIMD4(c,  s, 0, 0),
            SIMD4(-s, c, 0, 0),
            SIMD4(0,  0, 1, 0),
            SIMD4(0,  0, 0, 1)
        ))
    }

    func translationMatrix(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
        return simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0),
            SIMD4(0, 1, 0, 0),
            SIMD4(0, 0, 1, 0),
            SIMD4(x, y, z, 1)
        ))
    }
}
