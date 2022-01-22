package design_pattern.factory;

public class HighFactory implements AbstractFactory{
    @Override
    public Mask makeMask() {
        return new HighMask();
    }

    @Override
    public ProtectiveSuit makeProtectiveSuit() {
        return new HighProtectiveSuit();
    }
}
