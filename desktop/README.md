# Scriberr Desktop (macOS)

这是一个 Electron 桌面壳，内置 SSH 端口转发，用于把远端 Scriberr (`remote:8080`) 映射到本机端口并直接在桌面窗口中访问。

## 功能

- 可视化配置 SSH 隧道参数（主机、用户、端口、私钥、额外参数）
- 一键连接/断开 `ssh -L localPort:remoteHost:remotePort`
- 支持通过按钮将 SSH 配置保存到 JSON，并从 JSON 加载
- 连接后自动加载 `http://127.0.0.1:<localPort>`
- 支持打包 macOS `dmg`/`zip`

## 开发运行

```bash
cd desktop
npm install
npm run dev
```

## 打包 macOS

```bash
cd desktop
npm install
npm run dist:mac
```

产物在 `desktop/dist/`。

## 示例配置

- SSH Host: `ec2-35-173-125-142.compute-1.amazonaws.com`
- SSH User: `ubuntu`
- SSH Port: `22`
- Local Port: `18080`
- Remote Host: `127.0.0.1`
- Remote Port: `8080`

连接成功后桌面内访问地址即 `http://127.0.0.1:18080`。

## 注意事项

- 系统需要可用的 `ssh` 命令（macOS 默认自带）。
- 如果私钥有密码，请先在系统中配置 `ssh-agent`，否则后台隧道无法交互输入口令。
- 首次连接可能出现主机指纹确认，本项目默认设置 `StrictHostKeyChecking=accept-new`。
