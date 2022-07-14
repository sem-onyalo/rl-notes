import logging

def init_logger():
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )

def main():
    logging.info("hello, world")

if __name__ == "__main__":
    init_logger()

    main()

