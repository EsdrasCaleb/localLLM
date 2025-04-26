package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Accessories_toString_4_0_Test {

    @InjectMocks
    private Accessories accessories;

    @Mock
    private ArrayList<String> accessory;

    @BeforeEach
    public void setUp() {
        accessories = new Accessories();
    }

    @Test
    public void testToStringWithAccessories() {
        accessories.setAccessory(new String[] { "Accessory1", "Accessory2" });
        String expected = "# of Accessories = 2\n" + "MiniProduct - Accessory1\n" + "MiniProduct - Accessory2\n";
        assertEquals(expected, accessories.toString());
    }

    @Test
    public void testToStringWithNullAccessories() {
        accessories.setAccessory(null);
        String expected = "Accessories is null or size 0\n";
        assertEquals(expected, accessories.toString());
    }

    @Test
    public void testToStringWithEmptyAccessories() {
        accessories.setAccessory(new String[] {});
        String expected = "Accessories is null or size 0\n";
        assertEquals(expected, accessories.toString());
    }
}
