package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Accessories_toString_4_0_Test {

    @Test
    public void testToString_AccessoryListNotNull() {
        Accessories accessories = new Accessories();
        accessories.setAccessory(new String[] { "MiniProduct1", "MiniProduct2" });
        assertEquals("# of Accessories = 2\nMiniProduct - MiniProduct1\nMiniProduct - MiniProduct2", accessories.toString());
    }

    @Test
    public void testToString_AccessoryListNull() {
        Accessories accessories = new Accessories();
        assertEquals("Accessories is null or size 0", accessories.toString());
    }

    @Test
    public void testToString_AccessoryListEmpty() {
        Accessories accessories = new Accessories();
        assertEquals("Accessories is null or size 0", accessories.toString());
    }
}
