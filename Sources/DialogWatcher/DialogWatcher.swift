// 핵심 로직 — AXObserver로 모든 앱의 다이얼로그 창 감시 및 마우스 이동
import Cocoa
import ApplicationServices

class DialogWatcher {
    private var observers: [pid_t: AXObserver] = [:]
    private var runningAppsObserver: NSObjectProtocol?

    func startWatching() {
        // 현재 실행 중인 앱 전부 등록
        for app in NSWorkspace.shared.runningApplications {
            registerApp(app)
        }

        // 새로 실행되는 앱 감시
        runningAppsObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didLaunchApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] note in
            if let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication {
                self?.registerApp(app)
            }
        }

        NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didTerminateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] note in
            if let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication {
                self?.observers.removeValue(forKey: app.processIdentifier)
            }
        }
    }

    private func registerApp(_ app: NSRunningApplication) {
        let pid = app.processIdentifier
        guard pid > 0, observers[pid] == nil else { return }

        var observer: AXObserver?
        let callback: AXObserverCallback = { _, element, notification, refcon in
            guard let refcon else { return }
            let watcher = Unmanaged<DialogWatcher>.fromOpaque(refcon).takeUnretainedValue()
            watcher.handleWindowCreated(element: element, notification: notification as String)
        }

        guard AXObserverCreate(pid, callback, &observer) == .success,
              let observer else { return }

        let axApp = AXUIElementCreateApplication(pid)
        let selfPtr = Unmanaged.passUnretained(self).toOpaque()

        // 창 생성 / 시트 생성 알림 등록
        for notif in [
            kAXWindowCreatedNotification,
            kAXSheetCreatedNotification,
            kAXDrawerCreatedNotification
        ] {
            AXObserverAddNotification(observer, axApp, notif as CFString, selfPtr)
        }

        CFRunLoopAddSource(
            CFRunLoopGetMain(),
            AXObserverGetRunLoopSource(observer),
            .defaultMode
        )

        observers[pid] = observer
    }

    func handleWindowCreated(element: AXUIElement, notification: String) {
        // 잠깐 대기 — 창이 완전히 뜨기 전에 frame 쿼리하면 nil 반환
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            self.moveMouseToDialogIfNeeded(element: element)
        }
    }

    private func moveMouseToDialogIfNeeded(element: AXUIElement) {
        // role 확인
        var roleValue: CFTypeRef?
        AXUIElementCopyAttributeValue(element, kAXRoleAttribute as CFString, &roleValue)
        guard let role = roleValue as? String else { return }

        // AXSheet, AXDrawer, AXWindow(서브롤 AXDialog) 대상
        var isDialog = (role == kAXSheetRole || role == kAXDrawerRole)

        if !isDialog && role == kAXWindowRole {
            var subRoleValue: CFTypeRef?
            AXUIElementCopyAttributeValue(element, kAXSubroleAttribute as CFString, &subRoleValue)
            if let subRole = subRoleValue as? String, subRole == kAXDialogSubrole {
                isDialog = true
            }
        }

        guard isDialog else { return }

        // 창 위치·크기 가져오기
        var posValue: CFTypeRef?
        var sizeValue: CFTypeRef?
        AXUIElementCopyAttributeValue(element, kAXPositionAttribute as CFString, &posValue)
        AXUIElementCopyAttributeValue(element, kAXSizeAttribute as CFString, &sizeValue)

        guard let posValue, let sizeValue else { return }

        var position = CGPoint.zero
        var size = CGSize.zero
        AXValueGetValue(posValue as! AXValue, .cgPoint, &position)
        AXValueGetValue(sizeValue as! AXValue, .cgSize, &size)

        guard size.width > 0, size.height > 0 else { return }

        let center = CGPoint(
            x: position.x + size.width / 2,
            y: position.y + size.height / 2
        )

        // 제목 출력 (디버그용)
        var titleValue: CFTypeRef?
        AXUIElementCopyAttributeValue(element, kAXTitleAttribute as CFString, &titleValue)
        let title = (titleValue as? String) ?? "(제목 없음)"

        print("🪟 다이얼로그 감지: \"\(title)\" → 중심 (\(Int(center.x)), \(Int(center.y)))")

        moveMouse(to: center)
    }

    private func moveMouse(to point: CGPoint) {
        // macOS 좌표계: 스크린 좌상단 기준 (Quartz 좌표와 동일)
        guard let event = CGEvent(
            mouseEventSource: nil,
            mouseType: .mouseMoved,
            mouseCursorPosition: point,
            mouseButton: .left
        ) else { return }

        event.post(tap: .cghidEventTap)
    }
}
