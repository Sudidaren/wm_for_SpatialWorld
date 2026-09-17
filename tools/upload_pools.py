"""Upload the non-eval AI2-THOR + ProcTHOR pools to weste (for cloud training)."""
import os, sys, time, paramiko

HOST, PORT, PW = "connect.weste.seetacloud.com", 32808, "kQLLhVht3BEB"
JOBS = [("/mnt/d/lightwm_data_cov2", "/root/autodl-tmp/lightwm_data_cov2"),
        ("/mnt/d/lightwm_data_procthor2", "/root/autodl-tmp/lightwm_data_procthor2")]

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password=PW, timeout=30)
_, o, _ = c.exec_command("mkdir -p /root/autodl-tmp/lightwm_data_cov2 /root/autodl-tmp/lightwm_data_procthor2")
o.read()
sftp = c.open_sftp()

def ensure(remote_dir):
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try: sftp.stat(cur)
        except IOError:
            try: sftp.mkdir(cur)
            except IOError: pass

for local_root, remote_root in JOBS:
    if not os.path.isdir(local_root):
        continue
    ensure(remote_root)
    t0 = time.time(); nbytes = nskip = nup = 0
    for dp, _, fns in os.walk(local_root):
        for fn in fns:
            lp = os.path.join(dp, fn)
            rel = os.path.relpath(lp, local_root).replace(os.sep, "/")
            rp = f"{remote_root}/{rel}"
            size = os.path.getsize(lp)
            try:
                if sftp.stat(rp).st_size == size:
                    nskip += 1; nbytes += size; continue
            except IOError: pass
            ensure(os.path.dirname(rp))
            sftp.put(lp, rp); nup += 1; nbytes += size
            if nup % 200 == 0:
                print(f"  {os.path.basename(local_root)}: {nbytes/1e9:.2f} GB, "
                      f"{nup} uploaded / {nskip} skipped, {time.time()-t0:.0f}s", flush=True)
    print(f"{local_root} -> {remote_root}: {nbytes/1e9:.2f} GB "
          f"(uploaded {nup}, skipped {nskip}) in {time.time()-t0:.0f}s", flush=True)
sftp.close(); c.close()
