import subprocess
import time

procrastination_sites = ["netflix", "youtube", "twitch", "instagram", "facebook", "reddit", "twitter", "tumblr", "pinterest", "whatsapp", "snapchat",
                         "messenger", "telegram", "tiktok", "quora", "tinder", "grindr", "bumble", "hinge", "eharmony"]

# counter = 0

while True:
    result = subprocess.check_output('chrome-cli info', shell=True, text=True)
    time.sleep(2.5)
    result = subprocess.check_output('chrome-cli info', shell=True, text=True)
    
    ## printing the site you are on rn
    # print((result.split('\n'))[3][5:])
    for i in procrastination_sites:
        if i in result.split('\n')[3][5:]:
            # print("You are procrastinating on", i)
            print("slap")
    
    ## Uncomment out the counter line above to make this a timed loop for 50 seconds.
    ## Change the counter > 50 line to whatever number for the amount of time you want to run the loop.
    # if counter > 50:
    #     break
    # else:
    #     counter += 5