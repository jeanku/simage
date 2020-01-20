class Demo2():
    def say(self):
        # app()
        print('demo2.say')



class ge:

    def getnext(self):
        i = 0
        while i < 10:
            yield i
            i += 1

    def run(self):
        # d1 = [i for i in range(10)]
        # # print(d1)
        # # exit(0)

        d = self.getnext()
        # print(d)
        # exit(0)
        for i in d:
            print(i)

ge().run()