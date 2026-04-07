FROM node:20-slim
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --production

COPY server/ ./server/
COPY dist/ ./dist/
COPY config/ ./config/

ENV NODE_ENV=production
ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

CMD ["node", "server/index.js"]
