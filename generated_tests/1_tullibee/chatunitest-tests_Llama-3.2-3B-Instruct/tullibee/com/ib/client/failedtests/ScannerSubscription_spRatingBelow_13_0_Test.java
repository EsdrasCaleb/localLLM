// Test method
package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_spRatingBelow_13_0_Test {

    @Mock
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setup() {
        when(scannerSubscription.spRatingBelow()).thenReturn("");
    }

    @Test
    public void testSpRatingBelow() {
        // Arrange
        String expected = "expected value";
        when(scannerSubscription.spRatingBelow()).thenReturn(expected);
        // Act
        String actual = scannerSubscription.spRatingBelow();
        // Assert
        assertEquals(expected, actual);
    }

    @Test
    public void testSpRatingBelow_Null() {
        // Arrange
        when(scannerSubscription.spRatingBelow()).thenReturn(null);
        // Act and Assert
        assertThrows(NullPointerException.class, () -> scannerSubscription.spRatingBelow());
    }

    @Test
    public void testSpRatingBelow_InvalidValue() {
        // Arrange
        when(scannerSubscription.spRatingBelow()).thenReturn("invalid value");
        // Act and Assert
        assertThrows(IllegalArgumentException.class, () -> scannerSubscription.spRatingBelow());
    }

    @Test
    public void testSpRatingBelow_Empty() {
        scannerSubscription = new ScannerSubscription();
        String result = scannerSubscription.spRatingBelow();
        assertTrue(result.isEmpty());
    }
}
