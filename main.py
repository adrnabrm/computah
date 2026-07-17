from agent import DukeBot

def main():
    bot = DukeBot()
    transcript = bot.listen()
    print(transcript)
    print(bot.run(transcript))

if __name__ == "__main__":
    main()