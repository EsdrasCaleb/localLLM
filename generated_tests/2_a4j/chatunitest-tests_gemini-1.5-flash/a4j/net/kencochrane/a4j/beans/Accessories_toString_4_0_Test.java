package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Accessories_toString_4_0_Test {

    @Test
    void testToString_EmptyAccessories() {
        Accessories accessories = new Accessories();
        assertEquals("Accessories is null or size 0\n", accessories.toString());
    }

    @Test
    void testToString_NullAccessories() {
        Accessories accessories = new Accessories();
        try {
            Field accessoryField = Accessories.class.getDeclaredField("accessory");
            accessoryField.setAccessible(true);
            accessoryField.set(accessories, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set accessory to null: " + e.getMessage());
        }
        assertEquals("Accessories is null or size 0\n", accessories.toString());
    }

    @Test
    void testToString_NonEmptyAccessories() {
        Accessories accessories = new Accessories();
        accessories.setAccessory(new String[] { "Accessory1", "Accessory2" });
        String expected = "# of Accessories = 2\n" + "MiniProduct - Accessory1\n" + "MiniProduct - Accessory2\n";
        assertEquals(expected, accessories.toString());
    }

    @Test
    void testToString_SingleAccessory() {
        Accessories accessories = new Accessories();
        accessories.setAccessory(new String[] { "Accessory1" });
        String expected = "# of Accessories = 1\n" + "MiniProduct - Accessory1\n";
        assertEquals(expected, accessories.toString());
    }
}
