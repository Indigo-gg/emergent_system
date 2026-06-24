import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: '.',
  publicDir: false,
  server: {
    open: true,
    port: 5174,
  },
  build: {
    rollupOptions: {
      input: {
        'slime-mold': resolve(__dirname, 'slime-mold.html'),
        'ecosystem': resolve(__dirname, 'ecosystem.html'),
        explorer: resolve(__dirname, 'explorer.html'),
      },
    },
  },
});
