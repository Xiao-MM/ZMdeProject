package design_pattern.factory;

public class LowFactory implements AbstractFactory{
    @Override
    public Mask makeMask() {
        return new LowMask();
    }

    @Override
    public ProtectiveSuit makeProtectiveSuit() {
        return new LowProtectiveSuit();
    }
}
