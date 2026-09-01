FROM node:20-slim
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --production

COPY server/ ./server/
COPY dist/ ./dist/
COPY config/ ./config/

ENV NODE_ENV=production
ENV PORT=7860
# Chat-session logs must live on a writable path. The server resolves its
# default logs dir relative to the repo layout, which does not exist in this
# flattened image; point it at a writable location under /app (owned by node).
ENV CHAT_LOGS_DIR=/app/logs/chat_sessions
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

# Run as non-root. The node:20-slim base image ships an unprivileged "node"
# user (uid 1000); reuse it rather than creating another account.
RUN chown -R node:node /app
USER node

CMD ["node", "server/index.js"]
