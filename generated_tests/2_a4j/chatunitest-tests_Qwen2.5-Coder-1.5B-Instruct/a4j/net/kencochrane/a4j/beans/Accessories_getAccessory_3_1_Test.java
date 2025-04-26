package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Accessories_getAccessory_3_1_Test {

    @Mock
    private ArrayList<String> accessoryList;

    @InjectMocks
    private Accessories accessories;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetAccessoryWithValidIndex() {
        // Arrange
        when(accessoryList.get(2)).thenReturn("Laptop");
        int index = 2;
        // Act
        String result = accessories.getAccessory(index);
        // Assert
        assertEquals("Laptop", result);
    }

    @Test
    public void testGetAccessoryWithInvalidIndex() {
        // Arrange
        when(accessoryList.get(3)).thenReturn(null);
        int index = 3;
        // Act
        String result = accessories.getAccessory(index);
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetAccessoryWithEmptyList() {
        // Arrange
        when(accessoryList.isEmpty()).thenReturn(true);
        int index = 0;
        // Act
        String result = accessories.getAccessory(index);
        // Assert
        assertNull(result);
    }
}
