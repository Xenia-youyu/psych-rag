/**
 * 用本机 Chrome 的 CDP 协议截图。
 *
 * 为什么不用 agent-browser：它要下 500MB Chromium，而本机已经有 Chrome。
 * 为什么不用 --screenshot 命令行：那个只能截首屏，注入不了 localStorage，
 * 登录后的四个页面全都看不到。
 *
 * 用法：
 *   node scripts/shot.mjs <url> <out.png>
 * 环境变量：
 *   SHOT_TOKEN / SHOT_USER  注入 localStorage，模拟已登录
 */

import { spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
const PORT = 9333
const ORIGIN = 'http://127.0.0.1:5173'

const url = process.argv[2]
const outPath = process.argv[3] || 'shot.png'
const token = process.env.SHOT_TOKEN || ''
const user = process.env.SHOT_USER || '{}'
const waitMs = Number(process.env.SHOT_WAIT || 2200)

if (!url) {
  console.error('usage: node scripts/shot.mjs <url> <out.png>')
  process.exit(2)
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

const profile = path.join('D:/tmp', `chrome-shot-${Date.now()}`)
const chrome = spawn(
  CHROME,
  [
    '--headless=new',
    `--remote-debugging-port=${PORT}`,
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--hide-scrollbars',
    '--window-size=1500,940',
    `--user-data-dir=${profile}`,
    'about:blank',
  ],
  { stdio: 'ignore' },
)

let ws = null
let nextId = 0
const pending = new Map()

function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++nextId
    pending.set(id, { resolve, reject })
    ws.send(JSON.stringify({ id, method, params }))
    setTimeout(() => {
      if (pending.has(id)) {
        pending.delete(id)
        reject(new Error(`timeout: ${method}`))
      }
    }, 20000)
  })
}

async function findTarget() {
  for (let i = 0; i < 60; i++) {
    try {
      const response = await fetch(`http://127.0.0.1:${PORT}/json/list`)
      const targets = await response.json()
      const page = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl)
      if (page) return page
    } catch {
      // Chrome 还没起来
    }
    await sleep(200)
  }
  throw new Error('Chrome 调试端口没起来')
}

async function main() {
  const target = await findTarget()
  ws = new WebSocket(target.webSocketDebuggerUrl)
  await new Promise((resolve, reject) => {
    ws.onopen = resolve
    ws.onerror = () => reject(new Error('CDP 连接失败'))
  })
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data)
    if (message.id && pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id)
      pending.delete(message.id)
      if (message.error) reject(new Error(message.error.message))
      else resolve(message.result)
    }
  }

  await send('Page.enable')
  await send('Runtime.enable')

  // 先落在同源页面上，才允许写 localStorage
  await send('Page.navigate', { url: `${ORIGIN}/#/login` })
  await sleep(1200)

  if (token) {
    await send('Runtime.evaluate', {
      expression: `localStorage.setItem('psy_token', ${JSON.stringify(token)});
                   localStorage.setItem('psy_user', ${JSON.stringify(user)});`,
    })
    // 必须先 reload 再跳目标页：只改 hash 不会重新加载应用，
    // Pinia store 还停在"未登录"的初始状态，左下角会显示未登录。
    await send('Page.reload', { ignoreCache: false })
    await sleep(1400)

    const targetHash = (url.split('#')[1] || '').replace(/^\//, '')
    if (targetHash && targetHash !== 'login') {
      await send('Runtime.evaluate', {
        expression: `location.hash = ${JSON.stringify('#' + targetHash)}`,
      })
    }
  } else {
    await send('Page.navigate', { url })
  }

  await sleep(waitMs)

  const clickText = process.env.SHOT_CLICK_TEXT || ''
  if (clickText) {
    await send('Runtime.evaluate', {
      returnByValue: true,
      expression: `(function () {
        const btn = [...document.querySelectorAll('button')]
          .find((node) => node.textContent.trim().includes(${JSON.stringify(clickText)}));
        if (!btn) return 'not-found';
        btn.click();
        return 'clicked';
      })()`,
    })
    await sleep(Number(process.env.SHOT_AFTER || 2500))
  }

  const question = process.env.SHOT_QUESTION || ''
  if (question) {
    // 直接改 textarea.value 不会触发 Vue 的 v-model，必须用原型上的 setter
    // 再手动派发 input 事件——否则按钮一直是 disabled
    await send('Runtime.evaluate', {
      returnByValue: true,
      expression: `(function () {
        const ta = document.querySelector('.composer textarea');
        if (!ta) return 'no-textarea';
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
        setter.call(ta, ${JSON.stringify(question)});
        ta.dispatchEvent(new Event('input', { bubbles: true }));
        return 'typed';
      })()`,
    })
    await sleep(600)

    await send('Runtime.evaluate', {
      returnByValue: true,
      expression: `(function () {
        const btn = [...document.querySelectorAll('.composer-actions button')]
          .find((node) => node.textContent.includes('发送'));
        if (!btn) return 'no-button';
        if (btn.disabled) return 'disabled';
        btn.click();
        return 'sent';
      })()`,
    })

    await sleep(Number(process.env.SHOT_THINK || 22000))
  }

  const shot = await send('Page.captureScreenshot', {
    format: 'png',
    captureBeyondViewport: false,
  })
  fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true })
  fs.writeFileSync(outPath, Buffer.from(shot.data, 'base64'))
  console.log(`saved ${outPath}`)
}

main()
  .catch((error) => {
    console.error('FAIL', error.message)
    process.exitCode = 1
  })
  .finally(async () => {
    try {
      ws?.close()
    } catch {}
    chrome.kill()
    await sleep(400)
    try {
      fs.rmSync(profile, { recursive: true, force: true })
    } catch {}
  })
