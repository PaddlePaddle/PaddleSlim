import os
import sys
import shutil
import paddle
from paddle.utils.cpp_extension.extension_utils import CustomOpInfo, custom_write_stub


def main(so_path):
    suffix = ".so"
    assert so_path.endswith(suffix), f"{so_path} should end with {suffix}"
    resource = os.path.basename(so_path)[:-len(suffix)]

    install_dir = os.path.abspath(os.path.join(os.path.dirname(paddle.__file__), "..", resource))
    os.makedirs(install_dir, exist_ok=True)
    install_so_path = os.path.join(install_dir, f"{resource}_pd_{suffix}")
    shutil.copyfile(so_path, install_so_path)
    CustomOpInfo.instance().add(resource, os.path.basename(install_so_path), install_so_path)
    pyfile = os.path.join(install_dir, f"{resource}.py")
    custom_write_stub(f"{resource}{suffix}", pyfile) 
    init_file = os.path.join(install_dir, "__init__.py")
    with open(init_file, "w") as f:
        f.write("import paddle\n")
        f.write(f"from .{resource} import *")

    print(f"Installed in {install_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
