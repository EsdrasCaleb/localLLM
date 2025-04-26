package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Accessories_getAccessory_3_1_Test {

    @Test
    void testGetAccessory() {
        Accessories accessories = new Accessories();
        String[] accessory = { "Hat", "Shoes", "Belt" };
        accessories.setAccessory(accessory);
        // Test with valid index
        String[] expectedAccessory = { "Hat", "Shoes", "Belt" };
        assertEquals(expectedAccessory, accessories.getAccessory(0));
        // Test with out-of-bounds index
        String[] outOfBoundsAccessory = { "Hat", "Shoes", "Belt" };
        assertEquals(outOfBoundsAccessory, accessories.getAccessory(3));
        // Test with null accessory
        String[] nullAccessory = null;
        assertNull(accessories.getAccessory(4));
    }
}
