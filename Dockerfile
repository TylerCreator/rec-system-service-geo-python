FROM node:17-alpine
RUN apk update && apk add bash
# RUN apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python
# RUN python3 -m ensurepip
# RUN pip3 install --no-cache --upgrade pip setuptools
RUN apk add py3-scikit-learn py3-pip
RUN apk add py3-pip
# RUN apk add --no-cache python3 py3-pip
RUN pip3 install threadpoolctl
WORKDIR /usr/src/app
COPY package*.json ./
RUN npm install
COPY ./ ./
COPY ./wait-for-it.sh ./
RUN chmod +x ./wait-for-it.sh
EXPOSE 8080
CMD ["node", "app.js"]


# FROM node:17-alpine
# RUN apk update && apk add bash

# RUN apk add py3-scikit-learn py3-pip
# RUN apk add py3-pip

# RUN pip3 install threadpoolctl
# WORKDIR /usr/src/app
# COPY package*.json ./
# RUN npm install
# COPY ./ ./
# COPY ./wait-for-it.sh ./
# RUN chmod +x ./wait-for-it.sh
# EXPOSE 8080
# CMD ["npm", "start"]