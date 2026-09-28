// Electron 壳：加载 dist 静态产物，并把 dev 代理前缀转发到上游，
// 让安装版也能使用 HuggingFace / ModelScope 在线模型搜索。
const { app, BrowserWindow, protocol, net } = require('electron');
const fs = require('fs');
const path = require('path');

const DIST = path.join(__dirname, '..', 'dist');
const UPSTREAMS = {
  'hf-api': 'https://huggingface.co',
  'hf-file': 'https://huggingface.co',
  'ms-file': 'https://modelscope.cn',
};
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.map': 'application/json',
  '.txt': 'text/plain; charset=utf-8',
  '.wasm': 'application/wasm',
};

protocol.registerSchemesAsPrivileged([
  { scheme: 'app', privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true } },
]);

function safeDistFile(pathname) {
  const rel = decodeURIComponent(pathname).replace(/^\/+/, '') || 'index.html';
  const abs = path.normalize(path.join(DIST, rel));
  if (!abs.startsWith(DIST)) return null;
  return fs.existsSync(abs) && fs.statSync(abs).isFile() ? abs : null;
}

async function handleAppScheme(request) {
  const url = new URL(request.url);
  const pathname = decodeURIComponent(url.pathname);

  // dev 代理前缀 → 上游（在线模型搜索 / config 拉取）
  for (const [prefix, upstream] of Object.entries(UPSTREAMS)) {
    if (pathname === `/${prefix}` || pathname.startsWith(`/${prefix}/`)) {
      const rest = pathname.slice(prefix.length + 1);
      return net.fetch(upstream + rest + url.search, {
        method: request.method,
        headers: { accept: request.headers.accept || '*/*' },
      });
    }
  }

  // 静态文件
  const file = safeDistFile(pathname);
  if (file) {
    const body = fs.readFileSync(file);
    return new Response(body, {
      headers: { 'content-type': MIME[path.extname(file).toLowerCase()] || 'application/octet-stream' },
    });
  }
  return new Response('not found', { status: 404 });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1320,
    height: 900,
    minWidth: 960,
    minHeight: 640,
    title: 'LLM GPU Calculator',
    backgroundColor: '#f8fafc',
    autoHideMenuBar: true,
    webPreferences: { contextIsolation: true, nodeIntegration: false },
  });
  win.loadURL('app://bundle/');
  return win;
}

app.whenReady().then(() => {
  protocol.handle('app', handleAppScheme);
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
