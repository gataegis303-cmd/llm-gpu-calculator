import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import { inspectAttr } from 'kimi-plugin-inspect-react'
import { HttpsProxyAgent } from 'https-proxy-agent'

// 本机走系统代理时，让 vite 的外发请求（在线查询模型数据）也经由它
const outboundProxy =
  process.env.https_proxy || process.env.HTTPS_PROXY || process.env.http_proxy || process.env.HTTP_PROXY
const outboundAgent = outboundProxy ? new HttpsProxyAgent(outboundProxy) : undefined

// https://vite.dev/config/
export default defineConfig({
  base: './',
  plugins: [inspectAttr(), react()],
  server: {
    port: 3000,
    proxy: {
      // 在线自动查询：浏览器 → vite 代理 → 上游，绕开 CORS
      '/hf-api': {
        target: 'https://huggingface.co',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/hf-api/, ''),
        ...(outboundAgent ? { agent: outboundAgent } : {}),
      },
      '/hf-file': {
        target: 'https://huggingface.co',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/hf-file/, ''),
        ...(outboundAgent ? { agent: outboundAgent } : {}),
      },
      '/ms-file': {
        target: 'https://modelscope.cn',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/ms-file/, ''),
        ...(outboundAgent ? { agent: outboundAgent } : {}),
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
