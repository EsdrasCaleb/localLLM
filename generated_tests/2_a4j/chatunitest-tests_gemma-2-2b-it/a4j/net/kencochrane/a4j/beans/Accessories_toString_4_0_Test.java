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
    void testToString() {
        Accessories accessories = new Accessories();
        accessories.setAccessory(new String[] { "Apple", "Banana" });
        String expected = "# of Accessories = 2\nMiniProduct - Apple\nMiniProduct - Banana\n";
        String actual = accessories.toString();
        assertEquals(expected, actual);
    }
}
