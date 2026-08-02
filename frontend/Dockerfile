# Stage 1: Build
FROM node:22-slim AS builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ . .
RUN npm run build

# Stage 2: Runtime
FROM node:22-slim
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]