import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    tailwindcss(),
    react()
  ],
  server: {
    port: 3000, // Standard port for React app dev server
    host: '127.0.0.1',
    fs: {
      allow: ['.']
    },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000', // Proxy requests to Flask backend
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})

