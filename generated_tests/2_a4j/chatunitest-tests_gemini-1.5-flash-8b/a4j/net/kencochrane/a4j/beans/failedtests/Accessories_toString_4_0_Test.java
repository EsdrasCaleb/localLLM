package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Accessories_toString_4_0_Test {

    private Accessories accessories;

    @BeforeEach
    void setUp() {
        accessories = new Accessories();
    }

    @Test
    void testToStringEmptyAccessory() {
        assertEquals("Accessories is null or size 0\n", accessories.toString());
    }

    @Test
    void testToStringNullAccessory() {
        accessories.setAccessory(null);
        assertEquals("Accessories is null or size 0\n", accessories.toString());
    }

    @Test
    void testToStringWithAccessories() {
        String[] accessoriesArray = { "Accessory 1", "Accessory 2" };
        accessories.setAccessory(accessoriesArray);
        String expectedOutput = "# of Accessories = 2\n" + "MiniProduct - Accessory 1\n" + "MiniProduct - Accessory 2\n";
        assertEquals(expectedOutput, accessories.toString());
    }

    @Test
    void testToStringWithNullAccessoryElement() {
        String[] accessoriesArray = { "Accessory 1", null, "Accessory 3" };
        accessories.setAccessory(accessoriesArray);
        String expectedOutput = "# of Accessories = 3\n" + "MiniProduct - Accessory 1\n" + "MiniProduct - null\n" + "MiniProduct - Accessory 3\n";
        assertEquals(expectedOutput, accessories.toString());
    }
}
