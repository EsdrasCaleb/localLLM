package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Accessories_getAccessory_3_1_Test {

    @Test
    public void testGetAccessory() {
        Accessories accessories = new Accessories();
        accessories.setAccessory(new String[] { "apple", "banana", "cherry" });
        assertEquals("apple", accessories.getAccessory(0));
        assertEquals("banana", accessories.getAccessory(1));
        assertEquals("cherry", accessories.getAccessory(2));
    }
}
