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
    public void testGetAccessory_ValidIndex() {
        String result = accessories.getAccessory(1);
        assertEquals("Scarf", result);
    }

    @Test
    public void testGetAccessory_ValidIndexLastElement() {
        String result = accessories.getAccessory(2);
        assertEquals("Gloves", result);
    }

    @Test
    public void testGetAccessory_InvalidIndexTooHigh() {
        String result = accessories.getAccessory(3);
        assertNull(result);
    }

    @Test
    public void testGetAccessory_InvalidIndexNegative() {
        String result = accessories.getAccessory(-1);
        assertNull(result);
    }

    @Test
    public void testGetAccessory_EmptyList() throws Exception {
        Method setAccessoryMethod = Accessories.class.getDeclaredMethod("setAccessory", String[].class);
        setAccessoryMethod.setAccessible(true);
        setAccessoryMethod.invoke(accessories, (Object) new String[] {});
        String result = accessories.getAccessory(0);
        assertNull(result);
    }
}
