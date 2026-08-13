# Generated for Tmax agent bake. Placeholder: hamishi740/swerl-tmax-v3:8ea3a0258948
# Pre-bakes Node 22 + claude / opencode / pi / mini-swe-agent.
# Do NOT COPY test_sh or /tests (deferred verifier must stay out of the image).
FROM hamishi740/swerl-tmax-v3:8ea3a0258948

COPY assets/node22.tar /tmp_build/node22.tar
COPY assets/claude-code-local.tgz /tmp_build/claude-code-local.tgz
COPY assets/opencode-ai-local.tgz /tmp_build/opencode-ai-local.tgz
COPY assets/pi-coding-agent-local.tgz /tmp_build/pi-coding-agent-local.tgz
COPY assets/miniswe-wheels.tar /tmp_build/miniswe-wheels.tar

RUN set -eux; \
    mkdir -p /opt/node22; \
    tar xf /tmp_build/node22.tar -C /opt/node22 --strip-components=1; \
    ln -sfn /opt/node22/bin/node /usr/local/bin/node; \
    ln -sfn /opt/node22/bin/npm /usr/local/bin/npm; \
    ln -sfn /opt/node22/bin/npx /usr/local/bin/npx; \
    export PATH="/opt/node22/bin:/usr/local/bin:${PATH}"; \
    hash -r 2>/dev/null || true; \
    node --version; npm --version; \
    npm install -g --prefix=/usr/local --no-audit --no-fund /tmp_build/claude-code-local.tgz; \
    test -e /usr/local/bin/claude; \
    rm -f /tmp_build/node22.tar /tmp_build/claude-code-local.tgz

# OpenCode: unpack embedded binary (avoid npm double-free under Kaniko+proot).
RUN set -eux; \
    mkdir -p /opt/opencode; \
    tar xzf /tmp_build/opencode-ai-local.tgz -C /opt/opencode --strip-components=1; \
    rm -rf /usr/local/bin/opencode; \
    cp -f /opt/opencode/bin/opencode.exe /usr/local/bin/opencode; \
    chmod a+rx /usr/local/bin/opencode; \
    test -f /usr/local/bin/opencode; \
    ls -la /usr/local/bin/opencode; \
    rm -f /tmp_build/opencode-ai-local.tgz

RUN set -eux; \
    export PATH="/opt/node22/bin:/usr/local/bin:${PATH}"; \
    hash -r 2>/dev/null || true; \
    npm install -g --prefix=/usr/local --no-audit --no-fund /tmp_build/pi-coding-agent-local.tgz; \
    test -e /usr/local/bin/pi; \
    rm -f /tmp_build/pi-coding-agent-local.tgz

RUN set -eux; \
    if [ -x /opt/miniconda3/envs/testbed/bin/python3 ]; then \
      PY=/opt/miniconda3/envs/testbed/bin/python3; \
    else \
      PY=python3; \
    fi; \
    rm -rf /tmp_build/miniswe-wheels && mkdir -p /tmp_build/miniswe-wheels; \
    tar xf /tmp_build/miniswe-wheels.tar -C /tmp_build/miniswe-wheels; \
    "$PY" -m pip install --no-cache-dir --no-index \
      --find-links=file:///tmp_build/miniswe-wheels mini-swe-agent; \
    "$PY" -c 'from minisweagent.run.mini import app'; \
    MINI=$("$PY" -c 'import shutil; print(shutil.which("mini") or "")'); \
    test -n "$MINI"; \
    chmod a+x "$MINI"; \
    if [ "$MINI" != /usr/local/bin/mini ]; then \
      ln -sfn "$MINI" /usr/local/bin/mini; \
    fi; \
    chmod a+x /usr/local/bin/mini; \
    /usr/local/bin/mini --help >/dev/null; \
    id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent; \
    test -d /home/user; \
    chown -R agent:agent /home/agent; \
    rm -rf /tmp_build
