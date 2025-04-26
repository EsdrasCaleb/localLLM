package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Accessories_toString_4_2_Test {

    @Test
    public void testToString() {
        Accessories accessories = new Accessories();
        accessories.setAccessory(new String[] { "MiniProduct", "Laptop" });
        assertEquals(accessories.toString(), "# of Accessories = 2\nMiniProduct - MiniProduct\nLaptop - Laptop");
    }
}
