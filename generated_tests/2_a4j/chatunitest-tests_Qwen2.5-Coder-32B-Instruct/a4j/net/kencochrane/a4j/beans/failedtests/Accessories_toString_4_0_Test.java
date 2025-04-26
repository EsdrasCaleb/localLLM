package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Accessories_toString_4_0_Test {

    private Accessories accessories;

    @Mock
    private ArrayList<String> mockAccessoryList;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        accessories = new Accessories();
        // Use reflection to set the private 'accessory' field
        Field accessoryField = Accessories.class.getDeclaredField("accessory");
        accessoryField.setAccessible(true);
        accessoryField.set(accessories, mockAccessoryList);
    }

    @Test
    public void testToString_WithNullAccessoryList() throws Exception {
        // Set the accessory list to null
        Field accessoryField = Accessories.class.getDeclaredField("accessory");
        accessoryField.setAccessible(true);
        accessoryField.set(accessories, null);
        String result = accessories.toString();
        assertEquals("Accessories is null or size 0\n", result);
    }

    @Test
    public void testToString_WithEmptyAccessoryList() {
        when(mockAccessoryList.size()).thenReturn(0);
        String result = accessories.toString();
        assertEquals("Accessories is null or size 0\n", result);
    }

    @Test
    public void testToString_WithOneAccessory() {
        when(mockAccessoryList.size()).thenReturn(1);
        when(mockAccessoryList.get(0)).thenReturn("Product1");
        String result = accessories.toString();
        assertEquals("# of Accessories = 1\nMiniProduct - Product1\n", result);
    }

    @Test
    public void testToString_WithMultipleAccessories() {
        when(mockAccessoryList.size()).thenReturn(2);
        when(mockAccessoryList.get(0)).thenReturn("Product1");
        when(mockAccessoryList.get(1)).thenReturn("Product2");
        String result = accessories.toString();
        assertEquals("# of Accessories = 2\nMiniProduct - Product1\nMiniProduct - Product2\n", result);
    }

    @Test
    public void testToString_WithNullAccessory() {
        when(mockAccessoryList.size()).thenReturn(1);
        when(mockAccessoryList.get(0)).thenReturn(null);
        String result = accessories.toString();
        assertEquals("# of Accessories = 1\n", result);
    }
}
