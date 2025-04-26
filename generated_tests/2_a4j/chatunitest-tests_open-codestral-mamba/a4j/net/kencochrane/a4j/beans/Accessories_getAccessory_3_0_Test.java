package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Accessories_getAccessory_3_0_Test {

    private Accessories accessories;

    @BeforeEach
    public void setUp() throws Exception {
        accessories = Mockito.mock(Accessories.class);
        Field accessoryField = Accessories.class.getDeclaredField("accessory");
        accessoryField.setAccessible(true);
        accessoryField.set(accessories, new ArrayList<String>());
    }

    @Test
    public void testGetAccessoryWithinBounds() {
        ArrayList<String> accessoryList = new ArrayList<String>();
        accessoryList.add("Item1");
        accessoryList.add("Item2");
        accessoryList.add("Item3");
        try {
            Field accessoryField = Accessories.class.getDeclaredField("accessory");
            accessoryField.setAccessible(true);
            accessoryField.set(accessories, accessoryList);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        Mockito.when(accessories.getAccessory(1)).thenReturn("Item2");
        String result = accessories.getAccessory(1);
        assertEquals("Item2", result);
    }

    @Test
    public void testGetAccessoryOutOfBounds() {
        ArrayList<String> accessoryList = new ArrayList<String>();
        accessoryList.add("Item1");
        accessoryList.add("Item2");
        accessoryList.add("Item3");
        try {
            Field accessoryField = Accessories.class.getDeclaredField("accessory");
            accessoryField.setAccessible(true);
            accessoryField.set(accessories, accessoryList);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        Mockito.when(accessories.getAccessory(5)).thenReturn(null);
        String result = accessories.getAccessory(5);
        assertNull(result, "Index out of bounds should return null");
    }
}
