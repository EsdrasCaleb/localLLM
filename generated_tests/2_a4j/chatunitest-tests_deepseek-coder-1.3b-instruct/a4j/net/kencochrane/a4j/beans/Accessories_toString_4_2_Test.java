package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Accessories_toString_4_2_Test {

    private Accessories accessories;

    @BeforeEach
    public void setUp() {
        accessories = new Accessories();
        String[] accessoriesArray = { "Accessory1", "Accessory2", "Accessory3" };
        accessories.setAccessory(accessoriesArray);
    }

    @Test
    public void testToString() {
        String expectedOutput = "# of Accessories = 3\n" + "MiniProduct - Accessory1\n" + "MiniProduct - Accessory2\n" + "MiniProduct - Accessory3\n";
        assertEquals(expectedOutput, accessories.toString());
    }
}
