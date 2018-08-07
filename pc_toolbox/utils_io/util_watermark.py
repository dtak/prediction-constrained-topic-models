import pip

def make_string_of_reachable_modules_with_versions(context_dict=None):
    if context_dict is None:
        context_dict = globals()
    reachable_modules = dict()
    for key, val in context_dict.items():
        if key.startswith('_'):
            continue
        if str(type(val)).count('module'):
            # This trick will import parent package
            # e.g. scipy.stats becomes scipy
            if val.__package__ is None:
                mod_name = val.__name__
                mod = val
            else:
                try:
                    mod = __import__(val.__package__)
                except ImportError:
                    continue
                mod_name = mod.__name__
            reachable_modules[mod_name] = mod
            if hasattr(mod, '__requirements__'):
                for req_line in mod.__requirements__:
                    if req_line.count("=="):
                        mname = req_line.split("==")[0]
                    elif req_line.count(">="):
                        mname = req_line.split(">=")[0]
                    reachable_modules[mname] = None

    ver_info_list = [val for val in pip.operations.freeze.freeze()]

    explained_reachables = []
    ans_list = []
    for vstr in ver_info_list:
        if vstr.count('=='):
            name, version = vstr.split("==")
        elif vstr.count('egg'):
            parts = vstr.split('#egg=')
            name = parts[1]
            version = parts[0].replace('-e ', '')
            if version.count('.git@'):
                # Only display first 10 chars of git hash
                version = version[:version.find('.git@') + 15]
        else:
            name = vstr
        for mod_name in reachable_modules.keys():
            if vstr.count(mod_name):
                ans_list.append("%-40s %s" % (name, version))
                explained_reachables.append(mod_name)
    for rname, rmod in reachable_modules.items():
        if rname not in explained_reachables:
            if hasattr(rmod, '__version__'):
                version = rmod.__version__
                ans_list.append("%-40s %s" % (rname, version))
    # Sort and return a list
    ans_list = sorted([s for s in ans_list])
    ans = "\n".join(ans_list) + "\n"
    return ans