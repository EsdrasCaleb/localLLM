package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Accessories_getAccessory_3_4_Test {

    @Test
    public void testGetAccessory() {
        Accessories accessories = new Accessories();
        ArrayList<String> accessoryList = new ArrayList<>();
        accessoryList.add("Accessory1");
        accessoryList.add("Accessory2");
        accessories.setAccessory(accessoryList.toArray(new String[0]));
        String expected = "Accessory1";
        String actual = accessories.getAccessory(0);
        assertEquals(expected, actual);
    }
}
