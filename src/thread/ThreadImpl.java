package thread;

public class ThreadImpl extends Thread{
    @Override
    public void run() {
        System.out.println("Thread extend ...");
    }

    public static void main(String[] args) {
        new ThreadImpl().start();
    }
}
