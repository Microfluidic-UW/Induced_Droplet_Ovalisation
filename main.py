from libs.read_config import ReadConfig
def main() -> None:
    config_path = 'data/config/config.json'
    config = ReadConfig(config_path=config_path)
    config = config.read_config()
    print(config)

if __name__ == '__main__':
    main()