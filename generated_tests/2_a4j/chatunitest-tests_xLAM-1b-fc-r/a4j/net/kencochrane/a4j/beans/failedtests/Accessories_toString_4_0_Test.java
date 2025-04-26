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
    public void testToString() {
        Accessories accessories = new Accessories();
        accessories.setAccessory(new String[] { "accessory1", "accessory2", "accessory3" });
        String expectedOutput = "# of Accessories = 3\n" + "MiniProduct - accessory1\n" + "MiniProduct - accessory2\n" + "MiniProduct - accessory3\n";
        assertEquals(expectedOutput, accessories.toString());
    }

    @Test
    public void testToStringWithEmptyAccessory() {
        Accessories accessories = new Accessories();
        String expectedOutput = "# of Accessories = 0\n";
        assertEquals(expectedOutput, accessories.toString());
    }

    @Test
    public void testToStringWithNullAccessory() {
        Accessories accessories = new Accessories();
        accessories.setAccessory(null);
        String expectedOutput = "# of Accessories = 0\n";
        assertEquals(expectedOutput, accessories.toString());
    }
}
