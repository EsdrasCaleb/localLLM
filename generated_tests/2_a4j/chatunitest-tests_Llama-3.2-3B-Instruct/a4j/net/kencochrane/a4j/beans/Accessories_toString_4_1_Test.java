package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class Accessories_toString_4_1_Test {

    @Mock
    private Accessories accessories;

    @BeforeEach
    public void setup() {
        accessories = new Accessories();
    }

    @Test
    public void testToString_SingleAccessory_ReturnsCorrectString() {
        accessories.setAccessory(new String[] { "Accessory1" });
        String expected = "# of Accessories = 1\nMiniProduct - Accessory1\n";
        assertEquals(expected, accessories.toString());
    }

    @Test
    public void testToString_MultipleAccessories_ReturnsCorrectString() {
        accessories.setAccessory(new String[] { "Accessory1", "Accessory2", "Accessory3" });
        String expected = "# of Accessories = 3\nMiniProduct - Accessory1\nMiniProduct - Accessory2\nMiniProduct - Accessory3\n";
        assertEquals(expected, accessories.toString());
    }
}
