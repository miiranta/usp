# Tiles

SD + WEBDEV Project.

#### Setup and Run

- Create "environments" folder in "api/" and ".env" file inside it with the following content:

```typescript
MONGODB_CONNECTION_STRING = "mongodb://...";
JWT_SECRET = "senha_muito_secreta_e_segura";
PORT = 3000;
DOMAIN = "your-domain.com"; // or localhost
```

```typescript
// And if you  want to enable SSL, also add (PORT will be ignored):
SSL = "true";
EMAIL = "your-email@example.com";
```

- Then run the following commands in the terminal:

```bash
cd api
npm run setup
```

```bash
npm start
```

- Open your browser and navigate to `http://localhost:3000` (or https://{DOMAIN}) to see the application in action.

#### Setup and Run (Docker)

- Copy the **docs/docker-compose.example.yml** file to the root of your project (together with the Dockerfile) and rename it to **docker-compose.yml**.
- Edit the **docker-compose.yml** file to set your environment variables, such as `YOUR_MONGO_PASSWORD`, `JWT_SECRET`, `SSL`, `PORT`,`DOMAIN`, and `EMAIL`. Follow the comments in the file for guidance.
- > If SSL = 'false', edit **Dockerfile** and add your PORT to the `EXPOSE` line (e.g. PORT: 3000 `EXPOSE 80 443 3000`). Also edit the **docker-compose.yml** file to expose the same port (e.g. `3000:3000`).
- Run the following command in the terminal:

```bash
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml up
```

#### SSL warning
If you set `SSL = 'true'`:
- PORT variable will be ignored (443 will be used).
- The server will try to issue 2 certificates using the email you provided:

```
DOMAIN
www.DOMAIN
```
- Be sure to configure your DNS records for both!

#### Try it out!

- [tiles.luvas.io](https://tiles.luvas.io)

