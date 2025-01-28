from urllib.request import urlopen


def convert_str_to_bool(args):
    def str_to_bool(s):
        if not isinstance(s, str):
            return s
        if s.lower() == 'true':
            return True
        elif s.lower() == 'false':
            return False
        elif s.lower() == 'none':
            return None
        else:
            return s
    for key, value in vars(args).items():
        setattr(args, key, str_to_bool(value))

    return args


def check_offline():
    try:
        urlopen('https://www.google.com/', timeout=10)
        return False
    except:
        return True
