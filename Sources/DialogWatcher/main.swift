// 메인 진입점 — AXObserver로 다이얼로그 감시 후 마우스 이동
import Cocoa
import ApplicationServices
import Foundation

// launchd 환경에서 stdout 버퍼링 비활성화
setbuf(stdout, nil)

guard AXIsProcessTrusted() else {
    print("❌ Accessibility 권한 필요: 손쉬운 사용에서 이 앱을 허용하세요.")
    exit(1)
}

print("✅ Accessibility 권한 확인됨")
print("🔍 다이얼로그 감시 시작...")

let watcher = DialogWatcher()
watcher.startWatching()

// 런루프 실행 (앱 계속 대기)
RunLoop.main.run()
