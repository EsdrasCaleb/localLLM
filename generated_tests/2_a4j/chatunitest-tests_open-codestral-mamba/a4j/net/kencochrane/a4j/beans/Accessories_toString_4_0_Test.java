package net.kencochrane.a4j.beans;

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

    @BeforeEach
    public void setup() {
        accessories = mock(Accessories.class);
    }

    @Test
    public void testToString() {
        // Arrange
        when(accessories.getAccessoryArray()).thenReturn(new ArrayList<String>() {

            {
                add("MiniProduct1");
                add("MiniProduct2");
            }
        });
        // Act
        String result = accessories.toString();
        // Assert
        assertEquals("# of Accessories = 2\n" + "MiniProduct - MiniProduct1\n" + "MiniProduct - MiniProduct2\n", result);
    }
}
