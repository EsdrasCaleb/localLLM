package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Accessories_getAccessory_3_0_Test {

    @Test
    void testGetAccessoryValidIndex() {
        Accessories accessories = new Accessories();
        ArrayList<String> list = new ArrayList<>();
        list.add("A");
        list.add("B");
        list.add("C");
        accessories.setAccessory(list.toArray(new String[0]));
        String result = accessories.getAccessory(1);
        assertEquals("B", result);
    }

    @Test
    void testGetAccessoryInvalidIndex() {
        Accessories accessories = new Accessories();
        ArrayList<String> list = new ArrayList<>();
        list.add("A");
        list.add("B");
        accessories.setAccessory(list.toArray(new String[0]));
        String result = accessories.getAccessory(2);
        assertNull(result);
    }

    @Test
    void testGetAccessoryEmptyList() {
        Accessories accessories = new Accessories();
        accessories.setAccessory(new String[0]);
        String result = accessories.getAccessory(0);
        assertNull(result);
    }

    @Test
    void testGetAccessoryNegativeIndex() {
        Accessories accessories = new Accessories();
        ArrayList<String> list = new ArrayList<>();
        list.add("A");
        list.add("B");
        accessories.setAccessory(list.toArray(new String[0]));
        String result = accessories.getAccessory(-1);
        assertNull(result);
    }
}
