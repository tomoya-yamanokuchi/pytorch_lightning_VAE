import slackweb
import datetime

def elapsed_time_str(seconds):
    """秒をhh:mm:ss形式の文字列で返す
    （参照：https://imagingsolution.net/program/python/python-basic/elapsed_time_hhmmss/）

    Parameters
    ----------
    seconds : float
        表示する秒数

    Returns
    -------
    str
        hh:mm:ss形式の文字列
    """
    seconds = int(seconds + 0.5)    # 秒数を四捨五入
    h = seconds // 3600             # 時の取得
    m = (seconds - h * 3600) // 60  # 分の取得
    s = seconds - h * 3600 - m * 60 # 秒の取得

    return f"{h:02}:{m:02}:{s:02}"  # hh:mm:ss形式の文字列で返す



def notify_slack(web_hook_url: str, elapsed_time=None):
    assert type(web_hook_url) is str
    slack               = slackweb.Slack(web_hook_url)
    file_name           = __file__.split("/")[-1]
    time                = datetime.datetime.now()
    elapsed_time_format = elapsed_time_str(elapsed_time)
    text                = "Process is finished [{:}] --> {:} (elapsed_time: {:})"
    slack.notify(text=text.format(time, file_name, elapsed_time_format))