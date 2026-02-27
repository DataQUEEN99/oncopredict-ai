module.exports = {
  apps: [
    {
      name: 'oncopredict-ai',
      script: 'bash',
      args: '-c "streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"',
      cwd: '/home/user/webapp',
      env: {
        PYTHONUNBUFFERED: '1'
      },
      watch: false,
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      max_restarts: 10
    }
  ]
}
