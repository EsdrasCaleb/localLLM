package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class Accessories_getAccessory_3_0_Test {

    @ExtendWith(MockitoExtension.class)
    public class Accessories {

        private String[] accessory;

        public Accessories() {
            this.accessory = null;
        }

        public String getAccessory(int index) {
            if (accessory == null) {
                return null;
            }
            if (index < 0 || index >= accessory.length) {
                return null;
            }
            return accessory[index];
        }

        public void setAccessory(String[] accessory) {
            this.accessory = accessory;
        }
    }

    @Test
    public void testGetAccessory_EmptyList_ReturnsNull() {
        Accessories accessories = new Accessories();
        assertEquals(null, accessories.getAccessory(0));
    }

    @Test
    public void testGetAccessory_SingleElement_ReturnsElement() {
        Accessories accessories = new Accessories();
        String[] accessory = { "test" };
        accessories.setAccessory(accessory);
        assertEquals("test", accessories.getAccessory(0));
    }

    @Test
    public void testGetAccessory_OutOfRange_ReturnsNull() {
        Accessories accessories = new Accessories();
        assertEquals(null, accessories.getAccessory(1));
    }

    @Test
    public void testGetAccessory_LargeIndex_ReturnsNull() {
        Accessories accessories = new Accessories();
        String[] accessory = { "test", "test", "test" };
        accessories.setAccessory(accessory);
        assertEquals(null, accessories.getAccessory(3));
    }

    @Test
    public void testGetAccessory_MultipleElements_ReturnsCorrectElement() {
        Accessories accessories = new Accessories();
        String[] accessory = { "test", "test", "test" };
        accessories.setAccessory(accessory);
        assertEquals("test", accessories.getAccessory(1));
    }
}
