FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY app/package*.json ./app/
COPY api/package*.json ./api/

# Install Angular CLI
RUN npm install -g @angular/cli

# Install dependencies
WORKDIR /app/app
RUN npm install

WORKDIR /app/api
RUN npm install

# Copy source code
WORKDIR /app
COPY app/ ./app/
COPY api/ ./api/

# Create environment directories
RUN mkdir -p api/environments

EXPOSE 80 443 3000

# Run setup and start the application
WORKDIR /app/api
CMD ["sh", "-c", "npm run setup && node app.js"]
