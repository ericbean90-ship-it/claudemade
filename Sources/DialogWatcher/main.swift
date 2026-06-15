// 메인 진입점 — AXObserver로 다이얼로그 감시 후 마우스 이동
import Cocoa
import ApplicationServices

// Accessibility 권한 확인
func checkAccessibilityPermission() -> Bool {
    let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue(): true] as CFDictionary
    return AXIsProcessTrustedWithOptions(options)
}

guard checkAccessibilityPermission() else {
    print("❌ Accessibility 권한 필요: 시스템 환경설정 > 개인 정보 보호 > 손쉬운 사용에서 허용 후 재실행")
    exit(1)
}

print("✅ Accessibility 권한 확인됨")
print("🔍 다이얼로그 감시 시작...")

let watcher = DialogWatcher()
watcher.startWatching()

// 런루프 실행 (앱 계속 대기)
RunLoop.main.run()
