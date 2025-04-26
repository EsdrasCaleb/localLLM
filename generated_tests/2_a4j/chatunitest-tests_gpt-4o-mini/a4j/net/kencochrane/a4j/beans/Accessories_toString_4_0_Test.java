package net.kencochrane.a4j.beans;

import java.util.ArrayList;
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
    void testToString_WithNullAccessory() {
        // Test case where accessory is null
        String expected = "Accessories is null or size 0\n";
        assertEquals(expected, accessories.toString());
    }

    @Test
    void testToString_WithEmptyAccessory() {
        // Test case where accessory is empty
        accessories.setAccessory(new String[] {});
        String expected = "Accessories is null or size 0\n";
        assertEquals(expected, accessories.toString());
    }

    @Test
    void testToString_WithSingleAccessory() {
        // Test case with one accessory
        accessories.setAccessory(new String[] { "Accessory1" });
        String expected = "# of Accessories = 1\nMiniProduct - Accessory1\n";
        assertEquals(expected, accessories.toString());
    }

    @Test
    void testToString_WithMultipleAccessories() {
        // Test case with multiple accessories
        accessories.setAccessory(new String[] { "Accessory1", "Accessory2", "Accessory3" });
        String expected = "# of Accessories = 3\nMiniProduct - Accessory1\nMiniProduct - Accessory2\nMiniProduct - Accessory3\n";
        assertEquals(expected, accessories.toString());
    }
}
