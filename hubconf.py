from s3prl import hub

for _option in hub.options():
    # print(_option)
    globals()[_option] = getattr(hub, _option)
