package thread;

public class RunnableImpl implements Runnable{
    @Override
    public void run() {
        System.out.println(Thread.currentThread().getName() + "-Runnable...");

    }

    public static void main(String[] args) {
        new Thread(new RunnableImpl()).start();
    }
}
