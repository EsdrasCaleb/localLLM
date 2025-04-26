package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Accessories_getAccessory_3_0_Test {

    @InjectMocks
    private Accessories accessories;

    @Mock
    private ArrayList<String> mockAccessory;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        accessories.setAccessory(new String[] { "Watch", "Bag", "Shoes" });
    }

    @Test
    void testGetAccessoryValidIndex() {
        String result = accessories.getAccessory(1);
        assertEquals("Bag", result);
    }

    @Test
    void testGetAccessoryInvalidIndex() {
        String result = accessories.getAccessory(5);
        assertNull(result);
    }

    @Test
    void testGetAccessoryEmptyList() {
        accessories.setAccessory(new String[] {});
        String result = accessories.getAccessory(0);
        assertNull(result);
    }

    @Test
    void testGetAccessoryBoundaryIndex() {
        String result = accessories.getAccessory(2);
        assertEquals("Shoes", result);
    }

    @Test
    void testGetAccessoryNegativeIndex() {
        String result = accessories.getAccessory(-1);
        assertNull(result);
    }

    @Test
    void testGetAccessoryMockedList() {
        when(mockAccessory.size()).thenReturn(3);
        when(mockAccessory.get(1)).thenReturn("Bag");
        accessories.setAccessory(new String[] { "Watch", "Bag", "Shoes" });
        String result = accessories.getAccessory(1);
        assertEquals("Bag", result);
    }
}
