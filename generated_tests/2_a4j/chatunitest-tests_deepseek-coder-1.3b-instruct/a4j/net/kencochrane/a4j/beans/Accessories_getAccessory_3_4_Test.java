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

    private Accessories accessories;

    @BeforeEach
    public void setUp() {
        accessories = new Accessories();
        String[] accessoriesArray = { "Accessory1", "Accessory2", "Accessory3" };
        accessories.setAccessory(accessoriesArray);
    }

    @Test
    public void testGetAccessory() {
        int index = 1;
        String expected = "Accessory2";
        String actual = accessories.getAccessory(index);
        assertEquals(expected, actual);
    }
}
