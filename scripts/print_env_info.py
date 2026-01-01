import sys, site, json, os
info = {
    "python_executable": sys.executable,
    "python_version": sys.version.splitlines()[0],
    "site_packages": site.getsitepackages() if hasattr(site, 'getsitepackages') else [site.getusersitepackages()],
    "sys_path": sys.path[:10]
}
print(json.dumps(info, indent=2))
# Quick command to run:
# python scripts/print_env_info.py
