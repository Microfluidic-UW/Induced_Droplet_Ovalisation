from libs.read_video_and_analyse import ReadVideoAndAnalyse

def main() -> None:
    config_path = 'data/config/config.json'
    read_video_and_analyse = ReadVideoAndAnalyse(config_path)
    read_video_and_analyse.analyze_video()


if __name__ == '__main__':
    main()