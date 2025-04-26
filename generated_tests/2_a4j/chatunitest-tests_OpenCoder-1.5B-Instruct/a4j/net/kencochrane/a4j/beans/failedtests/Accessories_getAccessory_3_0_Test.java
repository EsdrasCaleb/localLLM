package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Accessories_getAccessory_3_0_Test {

    private Accessories accessories;

    @Test
    public void testGetAccessoryValidIndex() {
        accessories = new Accessories();
        accessories.setAccessory(new String[] { "Glasses", "Wallet", "Ring" });
        String expected = "Glasses";
        String actual = accessories.getAccessory(0);
        assertEquals(expected, actual);
    }

    @Test
    public void testGetAccessoryInvalidIndex() {
        accessories = new Accessories();
        accessories.setAccessory(new String[] { "Glasses", "Wallet", "Ring" });
        String expected = null;
        String actual = accessories.getAccessory(-1);
        assertEquals(expected, actual);
    }

    @Test
    public void testGetAccessoryOutOfBoundsIndex() {
        accessories = new Accessories();
        accessories.setAccessory(new String[] { "Glasses", "Wallet", "Ring" });
        String expected = null;
        String actual = accessories.getAccessory(3);
        assertEquals(expected, actual);
    }
}
