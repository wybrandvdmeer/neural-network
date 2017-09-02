package apps.sourcedetector;

public class Fifo {

    private byte [] bytes;
    private int position=0;

    public Fifo(int size) {
        bytes = new byte[size];
    }

    public void put(int token){
        put((byte)token);
    }

    public void put(byte token) {
        if(position < bytes.length) {
            bytes[position++] = token;
        } else {
            System.arraycopy(bytes, 1, bytes, 0, bytes.length - 1);
            bytes[bytes.length - 1] = token;
        }
    }

    public String toString() {
        return new String(bytes, 0, position);
    }

    public int getLength() {

        if(toString().length() < position) {
            return toString().length();
        }

        return position;
    }

    public boolean isFilled() {
        return !(position < bytes.length);
    }
}

