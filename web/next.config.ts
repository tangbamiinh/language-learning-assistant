import type { NextConfig } from "next";

/**
 * Next.js configuration for the Language Learning Assistant web app.
 *
 * Note on large request bodies (voice uploads):
 * Next.js API routes handle large JSON bodies by default.
 * If you encounter body size limits, you can:
 * - Use a custom server with Express/Koa and set `server.maxHttpHeaderSize` there
 * - Configure your reverse proxy (nginx/caddy) with `client_max_body_size`
 * - For Vercel deployments, request a limit increase from Vercel support
 */
const nextConfig: NextConfig = {
  /* config options here */
};

export default nextConfig;
