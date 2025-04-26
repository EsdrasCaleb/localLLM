package net.kencochrane.a4j.beans;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Accessories_getAccessory_3_0_Test {

    private Accessories accessories;

    @BeforeEach
    public void setUp() {
        accessories = new Accessories();
        accessories.setAccessory(new String[] { "Hat", "Scarf", "Gloves" });
    }

    @Test
    public void testGetAccessory_InvalidIndexNegative() {
        String result = accessories.getAccessory(-1);
        assertNull(result);
    }
}
