from time import time

if __name__ == "__main__":
    pass
    ## //--ARRAY/LIST SPEED TESTS--\\
    # sample_size = 10_000_000

    # np_sample = np.empty(sample_size)

    # start = time()
    # for i in range(1_000):
    #     np_sample = np.append(np_sample, i)
    # logger.info("TIMING: [{:,.20f}s]".format(time() - start))

    # np_sample = None

    # start = time()
    # list_sample = [0] * sample_size
    # # print(len(list_sample))
    # for i in range(1_000):
    #     list_sample.append(i)
    # logger.info("TIMING: [{:,.20f}s]".format(time() - start))

    # start = time()
    # array_sample = array.array("f", list_sample)
    # # print(len(array_sample))
    # for i in range(1_000):
    #     array_sample.append(i)
    # logger.info("TIMING: [{:,.20f}s]".format(time() - start))
