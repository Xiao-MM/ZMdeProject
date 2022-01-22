package thread;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ExecutorTest {
    public static void main(String[] args) {
        // CachedThreadPool：一个任务创建一个线程；创建一个可缓存线程池，如果线程池长度超过处理需要，可灵活回收空闲线程，若无可回收，则新建线程。
//        ExecutorService executorService = Executors.newCachedThreadPool();
//        for (int i = 0; i < 5; i++) {
//            executorService.execute(new RunnableImpl());
//        }
//        executorService.shutdown();

        // FixedThreadPool 创建一个定长线程池，可控制线程最大并发数，超出的线程会在队列中等待。
//        ExecutorService executorService = Executors.newFixedThreadPool(2);
//        for (int i = 0; i < 5; i++) {
//            executorService.execute(new RunnableImpl());
//        }
//        executorService.shutdown();
        ExecutorService executorService = Executors.newSingleThreadExecutor();
        for (int i = 0; i < 5; i++) {
            executorService.execute(new RunnableImpl());
        }
        executorService.shutdown();

    }
}
